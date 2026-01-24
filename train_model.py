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
from core_transformer import EinsumTransformer, TransformerConfig, EinsumOptimizer

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
    Returns (loss, grads_dict) where grads_dict is model._grads (copied) for this shard.
    """
    from core_transformer import EinsumTransformer, TransformerConfig
    cfg = TransformerConfig(config_dict)
    model = EinsumTransformer(cfg, _from_weights=weights_dict, _E_name=E_name, _layer_weights=layer_weights)
    logits = model.forward(x_shard, training=True, verbose=False)
    loss, d_logits = cross_entropy_loss(logits, y_shard)
    model.backward(d_logits)
    grads = {k: v.copy() for k, v in model._grads.items()}
    return loss, grads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/llama.json")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    
    cfg = TransformerConfig(config_dict)
    model = EinsumTransformer(cfg)
    optimizer = EinsumOptimizer(model, lr=args.lr)

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

    print(f"\n" + "#"*50)
    print(f"TRAINING START: {args.steps} steps" + (" (fixed batch)" if args.fix_batch else ""))
    print("#"*50)

    model._init_stats()  # So forward/backward accumulate across steps; tables printed at end

    # Generate data once if --fix-batch, else we'll generate each step
    if args.fix_batch:
        x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        y = np.roll(x, -1, axis=1)  # next-token target (learnable pattern)
        print("[INFO] Using fixed batch (loss should decrease if training works)")

    step_times = []
    losses = []
    for step in range(args.steps):
        t0 = time.time()

        if not args.fix_batch:
            x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
            y = np.roll(x, -1, axis=1)
            y[:, -1] = np.random.randint(0, cfg.vocab_size, (cfg.batch_size,))

        if ray_workers <= 1:
            # Single-process: forward, loss, backward, optimizer (timed separately)
            _t = time.time()
            logits = model.forward(x, training=True, verbose=False)
            loss, d_logits = cross_entropy_loss(logits, y)
            losses.append(loss)
            t_fwd = time.time() - _t
            _t = time.time()
            model.backward(d_logits)
            t_bwd = time.time() - _t
            _t = time.time()
            optimizer.step()
            t_opt = time.time() - _t
            t_fwd_bwd = None
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
            _t = time.time()
            optimizer.step()
            t_opt = time.time() - _t
            t_fwd, t_bwd = None, None

        dt = time.time() - t0
        step_times.append(dt)
        if ray_workers <= 1:
            print(f"[Step {step+1:02d}] Loss: {losses[-1]:.4f} | fwd: {t_fwd:.2f}s | bwd: {t_bwd:.2f}s | opt: {t_opt:.2f}s | total: {dt:.2f}s")
        else:
            print(f"[Step {step+1:02d}] Loss: {losses[-1]:.4f} | fwd+bwd: {t_fwd_bwd:.2f}s | opt: {t_opt:.2f}s | total: {dt:.2f}s")

    # Display final training summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    if step_times:
        total = sum(step_times)
        mean_st = np.mean(step_times)
        p50 = np.percentile(step_times, 50)
        p99 = np.percentile(step_times, 99)
        print(f"Total Steps:         {len(step_times)}")
        print(f"Total Time:          {total:.2f}s")
        print(f"Step time (fwd+bwd+opt):  mean={mean_st:.3f}s  p50={p50:.3f}s  p99={p99:.3f}s")
        if losses:
            print(f"Loss:                start={losses[0]:.4f}  end={losses[-1]:.4f}")
        
        if hasattr(model, '_memory_stats') and model._memory_stats.get('opt_per_step_mb'):
            avg_opt = np.mean(model._memory_stats['opt_per_step_mb'])
            print(f"Optimizer mem/step:  {avg_opt:.2f} MB (opt update only)")
        
        if hasattr(model, '_memory_stats') and 'grad_total_mb' in model._memory_stats:
            print(f"Gradient Memory:     {model._memory_stats['grad_total_mb']:.2f} MB")

        # Forward and backward timing tables (p50/p99 over all steps); skip when Ray workers run fwd/bwd
        if ray_workers <= 1:
            model.verbose = True
            model._print_stats_table("TIMING BREAKDOWN (seconds)")
            model._print_stats_table("BACKWARD PASS TIMING BREAKDOWN (seconds)")
    
    # Optional LoRA Merge
    if cfg.lora_merge_after_training:
        print("\n" + "="*50)
        print("LORA MERGE")
        print("="*50)
        model.merge_lora_weights()
    
    print("\n" + "#"*50)
    print("TRAINING COMPLETE")
    print("#" * 50)

if __name__ == "__main__":
    main()
