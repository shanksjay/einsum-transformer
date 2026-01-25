import os
import sys

# Pre-parse --num-threads and set BLAS/OpenMP env before any numpy (or numpy-using) import
for i, a in enumerate(sys.argv):
    if a == "--num-threads" and i + 1 < len(sys.argv):
        try:
            n = int(sys.argv[i + 1])
            os.environ["OMP_NUM_THREADS"] = str(n)
            os.environ["OPENBLAS_NUM_THREADS"] = str(n)
            os.environ["MKL_NUM_THREADS"] = str(n)
            if sys.platform == "darwin":
                os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
        except ValueError:
            pass
        break

import numpy as np
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from core_transformer import EinsumTransformer, TransformerConfig, EinsumOptimizer


class Tee:
    """Write to multiple streams (e.g. stdout and a log file)."""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return getattr(self.files[0], "isatty", lambda: False)()


def get_recommendations(args, losses, cfg, step_times, *, lr=None):
    """Return a list of recommendation strings to improve loss convergence. lr: effective learning rate (from --lr or config)."""
    lr_val = lr if lr is not None else (args.lr if args.lr is not None else 1e-4)
    recs = []
    if not losses:
        recs.append("- Run with more --steps to obtain a loss curve.")
        return recs
    start, end = losses[0], losses[-1]
    diff = end - start
    if diff > 0.01:
        recs.append(f"- Loss increased over training. Try: (1) lower --lr (current: {lr_val}), e.g. 5e-5 or 1e-5; (2) with --fix-batch, loss should decrease—if not, check implementation.")
    elif abs(diff) < 0.01:
        recs.append(f"- Loss is flat. Try: (1) higher --lr (e.g. 5e-4); (2) more --steps (current: {args.steps}); (3) --fix-batch to overfit one batch and verify the training loop.")
    else:
        batch = cfg.batch_size if cfg else "?"
        recs.append(f"- Loss decreased. For faster convergence: (1) more --steps; (2) learning-rate schedule (warmup + decay); (3) larger batch size if memory allows (current: {batch}).")
    if args.steps < 100:
        recs.append(f"- Consider more --steps (e.g. 100–1000) for meaningful convergence.")
    if args.fix_batch:
        recs.append("- Using --fix-batch: loss should decrease on this fixed batch; if it does not, reduce --lr or inspect weight init.")
    if lr_val > 0.01:
        recs.append(f"- LR={lr_val} is very high; if loss is flat or increasing, try --lr 1e-3 or 5e-4 to avoid overshooting.")
    return recs

def cross_entropy_loss(logits, targets):
    """
    logits: [B, T, V]
    targets: [B, T] (token ids)
    """
    B, T, V = logits.shape
    # Reshape for easier indexing
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Softmax
    probs = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    # Loss
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
    loss = -np.mean(np.log(correct_probs + 1e-9))
    
    # dLoss/dLogits
    d_logits_flat = probs.copy()
    d_logits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
    d_logits_flat /= len(targets_flat) # Mean reduction
    
    d_logits = d_logits_flat.reshape(B, T, V)
    return loss, d_logits


def _forward_backward_shard(weights_dict, config_dict, x_shard, y_shard, E_name, layer_weights):
    """
    Run forward + loss + backward on a batch shard. Used as a Ray remote.
    Returns (loss, grads_dict, t_fwd, t_bwd, fwd_mem, bwd_mem).
    """
    import time
    from core_transformer import EinsumTransformer, TransformerConfig
    cfg = TransformerConfig(config_dict)
    model = EinsumTransformer(cfg, _from_weights=weights_dict, _E_name=E_name, _layer_weights=layer_weights)
    _t = time.time()
    logits = model.forward(x_shard, training=True, verbose=False)
    t_fwd = time.time() - _t
    fwd_mem = getattr(model, '_memory_stats', {}).get('fwd_activation_mb', 0.0)
    loss, d_logits = cross_entropy_loss(logits, y_shard)
    _t = time.time()
    model.backward(d_logits)
    t_bwd = time.time() - _t
    bwd_mem = getattr(model, '_memory_stats', {}).get('grad_total_mb', 0.0)
    grads = {k: v.copy() for k, v in model._grads.items()}
    return loss, grads, t_fwd, t_bwd, fwd_mem, bwd_mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/llama.json")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate. Overrides config 'learning_rate' when set. If omitted, config learning_rate or 1e-4 is used.")
    parser.add_argument("--max-layers", type=int, default=None, 
                        help="Optional: limit number of layers for testing (uses full model by default)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--fix-batch", action="store_true",
                        help="Reuse the same batch every step (for overfit/debug; loss should decrease)")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="BLAS/OpenMP threads (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, VECLIB_MAXIMUM_THREADS on macOS). Set before numpy import. Example: --num-threads 8")
    parser.add_argument("--ray-workers", type=int, default=1,
                        help="Number of Ray workers for data-parallel forward/backward. 1=disabled. Requires optional 'ray' dependency.")
    args = parser.parse_args()

    log_file = None
    log_path = None
    _real_stdout = sys.stdout
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = log_dir / f"train_{ts}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write("=== Command ===\n")
        log_file.write(" ".join([sys.executable] + sys.argv) + "\n\n")
        log_file.write("=== Output ===\n")
        log_file.flush()
        sys.stdout = Tee(_real_stdout, log_file)

        if args.num_threads is not None:
            print(f"[INFO] BLAS/OpenMP threads: {args.num_threads} (set before numpy import)")
        if args.seed is not None:
            np.random.seed(args.seed)
            print(f"[INFO] Random seed: {args.seed}")

        with open(args.config, "r") as f:
            config_dict = json.load(f)
    
        # Optionally limit layers for testing
        if args.max_layers is not None:
            config_dict["n_layers"] = min(config_dict["n_layers"], args.max_layers)
            print(f"[INFO] Limiting to {config_dict['n_layers']} layers for testing")

        # Learning rate: --lr overrides config "learning_rate"; if --lr omitted, use config or 1e-4
        lr = args.lr if args.lr is not None else config_dict.get("learning_rate", 1e-4)

        cfg = TransformerConfig(config_dict)
        model = EinsumTransformer(cfg)
        optimizer = EinsumOptimizer(model, lr=lr)

        ray_workers = args.ray_workers
        if ray_workers > 1:
            try:
                import ray
            except ImportError:
                print("[ERROR] ray is required for --ray-workers>1. Install with: pip install ray")
                sys.exit(1)
            # Reduce Ray’s optional services and stderr noise: disable dashboard, usage stats, and
            # the GPU env-override FutureWarning. Set before ray.init().
            os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
            ray.init(include_dashboard=False)
            print(f"[INFO] Ray workers: {ray_workers} (data-parallel fwd/bwd)")
            _fwd_bwd_remote = ray.remote(_forward_backward_shard)

        lr_src = "from --lr" if args.lr is not None else "from config or default"
        print(f"\n" + "#"*50)
        print(f"TRAINING START: {args.steps} steps, lr={lr} ({lr_src})" + (" (fixed batch)" if args.fix_batch else ""))
        print("#"*50)
        if args.fix_batch:
            print("[INFO] Using fixed batch (loss should decrease if training works)")

        # Table header for training steps
        print("\n" + "="*95)
        print(f"| {'Step':<6} | {'Loss':<10} | {'Total (s)':<10} | {'FWD (s)':<10} | {'BWD (s)':<10} | {'OPT (s)':<10} | {'Peak Mem':<10} |")
        print("-" * 95)

        model._init_stats()  # So forward/backward accumulate across steps; tables printed at end

        # Generate data once if --fix-batch, else we'll generate each step
        if args.fix_batch:
            x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
            y = np.roll(x, -1, axis=1)  # next-token target (learnable pattern)

        step_times = []
        step_t_fwd, step_t_bwd, step_t_opt = [], [], []
        step_fwd_mem, step_bwd_mem = [], []
        losses = []
        for step in range(args.steps):
            t0 = time.time()

            if not args.fix_batch:
                x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
                y = np.roll(x, -1, axis=1)
                y[:, -1] = np.random.randint(0, cfg.vocab_size, (cfg.batch_size,))

            if ray_workers <= 1:
                # Single-process: forward, loss, backward, optimizer (timed and mem)
                _t = time.time()
                logits = model.forward(x, training=True, verbose=False)
                loss, d_logits = cross_entropy_loss(logits, y)
                losses.append(loss)
                t_fwd = time.time() - _t
                fwd_mem = getattr(model, '_memory_stats', {}).get('fwd_activation_mb', 0.0)
                _t = time.time()
                model.backward(d_logits)
                t_bwd = time.time() - _t
                bwd_mem = getattr(model, '_memory_stats', {}).get('grad_total_mb', 0.0)
                _t = time.time()
                optimizer.step()
                t_opt = time.time() - _t
                opt_mem = (model._memory_stats.get('opt_per_step_mb') or [0])[-1]
                t_fwd_bwd, r0 = None, None
            else:
                # Data-parallel: shard batch, run fwd+bwd in Ray workers, average grads, optimizer
                N = ray_workers
                B, T = x.shape[0], x.shape[1]
                sizes = [B // N] * N
                for i in range(B % N):
                    sizes[i] += 1
                ofs = 0
                x_shards, y_shards = [], []
                for s in sizes:
                    if s > 0:
                        x_shards.append(x[ofs : ofs + s])
                        y_shards.append(y[ofs : ofs + s])
                    ofs += s
                x_shards = [a for a in x_shards if a.size > 0]
                y_shards = [a for a in y_shards if a.size > 0]
                if not x_shards:
                    raise ValueError(f"batch_size={B} < ray_workers={N}: no shards. Use smaller --ray-workers or larger batch.")
                _t = time.time()
                weights_dict = model.get_weights_dict()
                weights_ref = ray.put(weights_dict)
                futures = [
                    _fwd_bwd_remote.remote(weights_ref, config_dict, x_i, y_i, model.E_name, model.layer_weights)
                    for x_i, y_i in zip(x_shards, y_shards)
                ]
                results = ray.get(futures)
                r0 = results[0]
                n_total = sum(xi.size for xi in x_shards)
                loss = sum(ri[0] * (xi.shape[0] * xi.shape[1]) for ri, xi in zip(results, x_shards)) / n_total
                losses.append(loss)
                model._grads = {}
                for k in results[0][1].keys():
                    model._grads[k] = sum(
                        (x_shards[i].shape[0] * x_shards[i].shape[1] / n_total) * results[i][1][k]
                        for i in range(len(results))
                    )
                t_fwd_bwd = time.time() - _t
                t_fwd = r0[2] if len(r0) > 2 else 0.0
                t_bwd = r0[3] if len(r0) > 3 else 0.0
                fwd_mem = r0[4] if len(r0) > 4 else 0.0
                # BWD: use driver's aggregated _grads size (comparable to non-Ray: grad memory on this process)
                bwd_mem = sum(g.nbytes for g in model._grads.values()) / (1024 * 1024)
                _t = time.time()
                optimizer.step()
                t_opt = time.time() - _t
                opt_mem = (model._memory_stats.get('opt_per_step_mb') or [0])[-1]

            dt = time.time() - t0
            step_times.append(dt)

            if ray_workers <= 1:
                t_f, t_b, t_o = t_fwd, t_bwd, t_opt
            else:
                t_f, t_b, t_o = t_fwd, t_bwd, t_opt
            step_t_fwd.append(t_f)
            step_t_bwd.append(t_b)
            step_t_opt.append(t_opt)
            step_fwd_mem.append(fwd_mem)
            step_bwd_mem.append(bwd_mem)
            tot = max(dt, 1e-9)
            pF = (t_f / tot) * 100
            pB = (t_b / tot) * 100
            pO = (t_opt / tot) * 100
            mem_peak = max(fwd_mem, bwd_mem, opt_mem)  # peak across FWD/BWD/OPT assuming prior-phase memory is freed

            if ray_workers > 1 and step == 0:
                # We'll print this as a note after the table starts if needed, but for now let's keep the table clean
                pass

            print(f"| {step+1:<6} | {losses[-1]:<10.4f} | {dt:<10.2f} | {t_f:<10.2f} | {t_b:<10.2f} | {t_o:<10.2f} | {mem_peak:<10.1f} |")

        print("="*95)

        # Display final training summary in the requested format
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)

        if losses:
            # Average loss delta across all steps
            total_delta = losses[-1] - losses[0]
            avg_delta = total_delta / len(losses) if len(losses) > 0 else 0
            print(f"Loss Trend(LR: {lr}):  start={losses[0]:.4f}  end={losses[-1]:.4f}  change per step = {avg_delta:.6f}")

        if step_times:
            print("\nExecution Time breakdown:")
            print("-" * 50)
            print(f"{'Phase':<25} | {'p50':<10} | {'p99':<10}")
            print("-" * 50)
            if step_t_fwd:
                p50_f = np.percentile(step_t_fwd, 50)
                p99_f = np.percentile(step_t_fwd, 99)
                print(f"Step time (fwd):          | {p50_f:<10.3f} | {p99_f:<10.3f}")
            if step_t_bwd:
                p50_b = np.percentile(step_t_bwd, 50)
                p99_b = np.percentile(step_t_bwd, 99)
                print(f"Step time (bwd):          | {p50_b:<10.3f} | {p99_b:<10.3f}")
            if step_t_opt:
                p50_o = np.percentile(step_t_opt, 50)
                p99_o = np.percentile(step_t_opt, 99)
                print(f"Step time (opt):          | {p50_o:<10.3f} | {p99_o:<10.3f}")

            print("-" * 50)
            sum_p50 = ((np.percentile(step_t_fwd, 50) if step_t_fwd else 0) + \
                       (np.percentile(step_t_bwd, 50) if step_t_bwd else 0) + \
                       (np.percentile(step_t_opt, 50) if step_t_opt else 0)) * len(step_times)
            sum_p99 = ((np.percentile(step_t_fwd, 99) if step_t_fwd else 0) + \
                       (np.percentile(step_t_bwd, 99) if step_t_bwd else 0) + \
                       (np.percentile(step_t_opt, 99) if step_t_opt else 0)) * len(step_times)
            print(f"Total ({len(step_times)} steps)            | {sum_p50:<10.3f} | {sum_p99:<10.3f}")
            print("-" * 50)

            print("\nMemory Breakdown:")
            print("-" * 50)
            fwd_m = np.mean(step_fwd_mem) if step_fwd_mem else 0
            bwd_m = np.mean(step_bwd_mem) if step_bwd_mem else 0
            opt_m = np.mean(model._memory_stats.get('opt_per_step_mb', [0]))

            print(f"FWD mem/step:       {fwd_m:.2f} MB")
            print(f"BWD mem/step:       {bwd_m:.2f} MB")
            print(f"Optimizer mem/step: {opt_m:.2f} MB")

            # Realistic description of baseline memory (Weight Manager Cache)
            misc_mem = model.weights.get_cache_memory_mb() if hasattr(model.weights, 'get_cache_memory_mb') else 0
            print(f"Misc (baseline memory i.e. Weight Manager Cache) : {misc_mem:.2f} MB")

            # Instantaneous peak = Max(FWD, BWD, OPT) + Misc
            peak_comp = max(max(step_fwd_mem) if step_fwd_mem else 0,
                             max(step_bwd_mem) if step_bwd_mem else 0,
                             max(model._memory_stats['opt_per_step_mb']) if model._memory_stats.get('opt_per_step_mb') else 0)
            peak_total = peak_comp + misc_mem
            print(f"Total memory (Peak): {peak_total:.2f} MB")
            print("-" * 50)

            # Forward and backward timing tables (p50/p99 over all steps); skip when Ray workers run fwd/bwd
            if ray_workers <= 1:
                model.verbose = True
                model._print_stats_table("FWD per step breakdown (Seconds)")
                model._print_stats_table("BWD per step breakdown (Seconds)")
                model._print_stats_table("OPT per step breakdown (Seconds)")

        # Optional LoRA Merge
        if cfg.lora_merge_after_training:
            print("\n" + "="*50)
            print("LORA MERGE")
            print("="*50)
            model.merge_lora_weights()

        print("\n" + "#"*50)
        print("TRAINING COMPLETE")
        print("#" * 50)

        recs = get_recommendations(args, losses, cfg, step_times, lr=lr)
        print("\n=== Recommendations to improve loss convergence ===\n")
        for r in recs:
            print(r)
    finally:
        if log_file is not None:
            sys.stdout = _real_stdout
            log_file.close()
            print(f"[INFO] Run log saved to {log_path}")

if __name__ == "__main__":
    main()
