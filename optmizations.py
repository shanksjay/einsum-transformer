def enable_gqa(cfg, kv_heads):
    cfg["use_gqa"] = True
    cfg["n_kv_heads"] = kv_heads
    return cfg

def enable_mqa(cfg):
    cfg["use_mqa"] = True
    cfg["n_kv_heads"] = 1
    return cfg

def enable_moe(cfg, num_experts=8, top_k=2):
    cfg["use_moe"] = True
    cfg["num_experts"] = num_experts
    cfg["top_k"] = top_k
    return cfg

def enable_quantization(cfg, mode="W8A16"):
    """Enable mixed-precision quantization. mode: W8A16, W8A8, W4A16, W4A4, W8A4, etc."""
    cfg["quantized"] = True
    cfg["quantization"] = mode
    return cfg

def enable_prefill_decode_split(cfg):
    cfg["prefill_only"] = True
    return cfg

def enable_speculative_decoding(cfg, draft_layers=2):
    cfg["speculative"] = True
    cfg["draft_layers"] = draft_layers
    return cfg

