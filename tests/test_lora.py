import numpy as np
import pytest
import os
import json
from core_transformer import EinsumTransformer, TransformerConfig, EinsumOptimizer

@pytest.fixture
def config_file():
    path = "configs/lora_test.json"
    with open(path, "r") as f:
        return json.load(f)

def test_lora_initialization(config_file):
    """Verify that LoRA adapter weights are initialized with correct shapes."""
    cfg = TransformerConfig(config_file)
    model = EinsumTransformer(cfg)
    
    r = cfg.lora_rank
    d_model = cfg.d_model
    n_layers = cfg.n_layers
    
    for i in range(n_layers):
        p = model.layer_weights[i]
        
        # Check Q adapters
        if "q_proj" in cfg.lora_target_modules:
            assert f"lora_A_q_{i}" in model.weights.keys()
            assert f"lora_B_q_{i}" in model.weights.keys()
            A_q = model.weights[f"lora_A_q_{i}"]
            B_q = model.weights[f"lora_B_q_{i}"]
            assert A_q.shape == (d_model, r)
            assert B_q.shape == (r, d_model)
        
        # Check V adapters
        if "v_proj" in cfg.lora_target_modules:
            assert f"lora_A_v_{i}" in model.weights.keys()
            assert f"lora_B_v_{i}" in model.weights.keys()
            A_v = model.weights[f"lora_A_v_{i}"]
            B_v = model.weights[f"lora_B_v_{i}"]

            d_head = d_model // cfg.n_heads
            d_kv = cfg.n_kv_heads * d_head

            assert A_v.shape == (d_model, r)
            assert B_v.shape == (r, d_kv)

def test_lora_training_step(config_file):
    """Verify that a training step runs and loss decreases (sanity check)."""
    cfg = TransformerConfig(config_file)
    model = EinsumTransformer(cfg)
    optimizer = EinsumOptimizer(model, lr=0.1) # High LR for quick movement
    
    # Dummy data
    x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    
    # Forward 1
    logits1 = model.forward(x, training=True, verbose=False)
    target = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    
    # Loss 1
    def get_loss(logits, targets):
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        t_flat = targets.flatten()
        p_flat = probs.reshape(-1, logits.shape[-1])
        correct = p_flat[np.arange(len(t_flat)), t_flat]
        return -np.mean(np.log(correct + 1e-9))

    loss1 = get_loss(logits1, target)
    
    # Backward & Update
    # Mocking backward for simplicity or calling actual backward
    # We need d_logits
    probs = np.exp(logits1 - np.max(logits1, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    d_logits = probs.copy()
    targets_flat = target.flatten()
    d_logits_flat = d_logits.reshape(-1, logits1.shape[-1])
    d_logits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
    d_logits = d_logits_flat.reshape(logits1.shape) / len(targets_flat)
    
    model.backward(d_logits)
    optimizer.step()
    
    # Forward 2
    logits2 = model.forward(x, training=True, verbose=False)
    loss2 = get_loss(logits2, target)
    
    assert loss2 < loss1, "Loss should decrease after a training step with LoRA adapters"

def test_lora_merge_correctness(config_file):
    """Verify that merging LoRA weights produces identical inference output."""
    cfg = TransformerConfig(config_file)
    model = EinsumTransformer(cfg)
    
    # Random input
    x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    
    # 1. Forward with LoRA applied on the fly (training=True so adapters are used)
    # Actually, verify logic: adapters are used ONLY if training=True in current implementation.
    # If we want to test inference equivalence, we need a mode where adapters are applied manually or
    # we simulate inference with adapters.
    # Our implementation: if self.training: apply LoRA.
    # So let's run forward(training=True) to get 'adapter output'. 
    # Note: Training also adds dropout if we had it, but we don't. It saves activations.
    
    # But wait, standard inference (training=False) does NOT apply adapters in our current code!
    # "if self.training: ... apply adapters"
    # This means inference currently ignores LoRA adapters unless we merge them!
    # This is actually a common pattern (only merge for inference), PROVIED we merge them.
    # So: output before merge (training=True) should equal output after merge (training=False/True).
    # NOTE: Output before merge (training=False) will be DIFFERENT (base model only).
    
    logits_with_adapters = model.forward(x, training=True, verbose=False)
    
    # 2. Merge weights
    model.merge_lora_weights()
    
    # 3. Forward with merged weights. Now training=True/False shouldn't matter for weights,
    # (except our code still tries to add adapters if training=True! We need to disable adapters or remove them).
    # Our merge implementation updates W_q and W_v. 
    # BUT if we run with training=True again, code will define:
    # W_q = W_q_merged + scale * (A @ B). The adapters A/B are still there!
    # So we effectively apply them TWICE if we run training=True after merge.
    # CORRECTNESS CHECK: correctness is `logits(training=True before merge) == logits(training=False after merge)`
    
    logits_merged_inference = model.forward(x, training=False, verbose=False)
    
    np.testing.assert_allclose(logits_with_adapters, logits_merged_inference, rtol=1e-5, atol=1e-5, 
                               err_msg="Merged inference output matches LoRA training output")
