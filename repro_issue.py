import numpy as np
import copy
from core_transformer import TransformerConfig, EinsumTransformer

def reproduce_issue():
    # Mimic the config causing issues: Llama 3.2 3B settings with LoRA and Quantization
    cfg_dict = {
        "arch": "llama",
        "d_model": 128, # Small for repro
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_ff": 256,
        "vocab_size": 1000,
        "batch_size": 1,
        "seq_len": 32,
        "max_new_tokens": 5,
        "weight_dtype": "bfloat16", # Trigger bfloat16 paths
        "quantization": "W8A16", # Trigger quantization
        "use_lora": True,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 8,
        "lora_alpha": 16.0
    }

    cfg = TransformerConfig(cfg_dict)

    print("Initializing model...")
    # This will init random weights, including LoRA
    model = EinsumTransformer(cfg)

    print("Running optimization...")
    try:
        # Trigger the optimized path logic manually to catch warnings
        model._optimize_for_inference()
        print("Optimization successful.")
    except Exception as e:
        print(f"Optimization FAILED: {e}")

    print("Checking weights...")
    # Check for NaNs
    for i, l in enumerate(model.optimized_layers):
        if np.isnan(l['W_qkv']).any() or np.isinf(l['W_qkv']).any():
            print(f"Layer {i} W_qkv contains NaNs or Infs!")
        else:
            print(f"Layer {i} W_qkv OK.")

if __name__ == "__main__":
    reproduce_issue()
