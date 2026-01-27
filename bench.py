import time
import numpy as np
import cProfile
import pstats
import argparse
import json
import sys
import copy
from core_transformer import TransformerConfig, EinsumTransformer

def benchmark(model, tokens, max_new_tokens=20, profile=False, name="Baseline"):
    # Warmup
    print(f"[{name}] Warming up...")
    try:
        model.clear_cache()
        # Need matching batch size for warmup
        warmup_tokens = tokens[:, :2]
        # Temporarily reduce max_new_tokens for warmup
        orig_max = model.cfg.max_new_tokens
        model.cfg.max_new_tokens = 2
        model.generate(warmup_tokens)
        model.cfg.max_new_tokens = orig_max
    except Exception as e:
        print(f"Warmup failed: {e}")

    if hasattr(model, "clear_cache"): model.clear_cache()

    print(f"[{name}] Starting benchmark: Batch={tokens.shape[0]}, SeqLen={tokens.shape[1]}, Gen={max_new_tokens}")

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    t0 = time.perf_counter()

    # Set max_new_tokens
    model.cfg.max_new_tokens = max_new_tokens
    generated = model.generate(tokens)

    t_end = time.perf_counter()

    if profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumtime')
        ps.print_stats(20)

    total_time = t_end - t0

    # Estimate decode time
    model.clear_cache()
    t_p_start = time.perf_counter()
    model.forward(tokens, inference=True, verbose=False)
    t_prefill = time.perf_counter() - t_p_start

    decode_time = total_time - t_prefill
    steps_per_sec = max_new_tokens / decode_time if decode_time > 0 else 0
    batch_size = tokens.shape[0]
    agg_tokens_per_sec = steps_per_sec * batch_size

    print(f"\n--- {name} Results ---")
    print(f"Prefill Latency: {t_prefill*1000:.2f} ms")
    print(f"Total Generation Time: {total_time:.4f} s")
    print(f"Decode Steps/s (Latency): {steps_per_sec:.2f}")
    print(f"Aggregate Tokens/s (Throughput): {agg_tokens_per_sec:.2f}")

    return agg_tokens_per_sec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fast_lora.json")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = json.load(f)

    cfg_dict['max_new_tokens'] = 50
    # Speculative settings for optimized model
    cfg_dict['speculative'] = False

    cfg = TransformerConfig(cfg_dict)

    # 1. Baseline
    print("\nInitializing Baseline Model...")
    baseline_model = EinsumTransformer(cfg)
    # Use shorter prompt to allow generation within seq_len limit
    prompt_len = min(32, cfg.seq_len // 2)
    tokens = np.random.randint(0, cfg.vocab_size, size=(cfg.batch_size, prompt_len))

    benchmark(baseline_model, tokens, max_new_tokens=cfg.max_new_tokens, profile=False, name="Baseline")

    # 2. Optimized (Standard)
    print("\nInitializing Optimized Model (Standard)...")
    # Standard EinsumTransformer handles optimizations internally now
    # We reuse weights to simulate 'inference from loaded model'
    opt_model = EinsumTransformer(cfg, _from_weights=baseline_model.weights, _E_name=baseline_model.E_name, _layer_weights=baseline_model.layer_weights)
    benchmark(opt_model, tokens, max_new_tokens=cfg.max_new_tokens, profile=args.profile, name="Optimized-Standard")

    # 3. Optimized (Speculative)
    print("\nInitializing Optimized Model (Speculative)...")
    cfg_spec = copy.deepcopy(cfg)
    cfg_spec.speculative = True
    cfg_spec.draft_layers = 2

    opt_model_spec = EinsumTransformer(cfg_spec, _from_weights=baseline_model.weights, _E_name=baseline_model.E_name, _layer_weights=baseline_model.layer_weights)
    benchmark(opt_model_spec, tokens, max_new_tokens=cfg.max_new_tokens, profile=args.profile, name="Optimized-Speculative")

    # 4. Optimized (LoRA + Speculative)
    print("\nInitializing Optimized Model (LoRA + Speculative)...")
    cfg_lora = copy.deepcopy(cfg)
    cfg_lora.speculative = True
    cfg_lora.draft_layers = 2
    cfg_lora.use_lora = True # Enable LoRA to test fallback behavior

    # We need to make sure weights contain LoRA params.
    # Baseline random init might have them if configured, but let's check.
    # config/fast_lora.json has use_lora: false but lora_target_modules defined.
    # EinsumTransformer init will create them if missing when use_lora=True is passed?
    # No, _from_weights skips init_weights_random/load_weights.
    # It calls _bind_weights.
    # If weights are missing, it might error or init them if code allows.
    # Let's ensure baseline has them.

    # Actually, let's just create a new model for this test to ensure weights exist
    opt_model_lora = EinsumTransformer(cfg_lora)
    # Use small batch for LoRA test to avoid OOM if random init is large
    tokens_lora = np.random.randint(0, cfg.vocab_size, size=(cfg.batch_size, prompt_len))
    benchmark(opt_model_lora, tokens_lora, max_new_tokens=cfg.max_new_tokens, profile=args.profile, name="Optimized-LoRA-Speculative")
