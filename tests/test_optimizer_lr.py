"""
Test that the optimizer correctly applies the learning rate to gradients.
Verifies: (1) weights change after step, (2) update follows W_new = W_old - lr * grad.
"""
import numpy as np
from core_transformer import EinsumTransformer, TransformerConfig, EinsumOptimizer


def _minimal_config():
    return {
        "arch": "llama",
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "d_ff": 64,
        "vocab_size": 256,
        "rope_base": 10000.0,
        "batch_size": 2,
        "seq_len": 8,
        "use_gqa": False,
        "use_moe": False,
        "use_lora": False,
        "quantization": "none",
        "weight_file": None,
        "hf_repo_id": None,
    }


def _cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    probs = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    correct = probs[np.arange(len(targets_flat)), targets_flat]
    loss = -np.mean(np.log(correct + 1e-9))
    d_logits_flat = probs.copy()
    d_logits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
    d_logits_flat /= len(targets_flat)
    return loss, d_logits_flat.reshape(B, T, V)


def test_optimizer_applies_lr_weight_change():
    """Weights must change after optimizer.step() when gradients are non-zero."""
    cfg = TransformerConfig(_minimal_config())
    model = EinsumTransformer(cfg)
    model._init_stats()
    lr = 0.1
    optimizer = EinsumOptimizer(model, lr=lr)

    x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    y = np.roll(x, -1, axis=1)
    y[:, -1] = 0

    logits = model.forward(x, training=True, verbose=False)
    loss, d_logits = _cross_entropy_loss(logits, y)
    model.backward(d_logits)

    # Pick a weight that uses simple update (no transpose): norm1_0
    key = "norm1_0"
    assert key in model._grads, f"Expected grad for {key}"
    old_w = np.array(model.weights[key], copy=True)
    grad = model._grads[key]
    optimizer.step()
    new_w = np.array(model.weights[key], copy=True)

    # W_new = W_old - lr * grad (no transpose for norm)
    expected = old_w - lr * grad
    np.testing.assert_allclose(new_w, expected, rtol=1e-5, atol=1e-5,
        err_msg="norm1_0 update must follow W_new = W_old - lr * grad")

    # Weight must have changed (grad is non-zero for a nontrivial batch)
    assert np.any(old_w != new_w), "Weights must change after step when grad is non-zero"


def test_optimizer_fix_batch_loss_decreases():
    """With --fix-batch (same batch every step), loss should decrease over several steps (single process)."""
    cfg = TransformerConfig(_minimal_config())
    model = EinsumTransformer(cfg)
    model._init_stats()
    optimizer = EinsumOptimizer(model, lr=0.05)

    np.random.seed(42)
    x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    y = np.roll(x, -1, axis=1)
    y[:, -1] = np.random.randint(0, cfg.vocab_size, (cfg.batch_size,))

    losses = []
    for _ in range(15):
        logits = model.forward(x, training=True, verbose=False)
        loss, d_logits = _cross_entropy_loss(logits, y)
        losses.append(loss)
        model.backward(d_logits)
        optimizer.step()

    # Loss should be lower at the end than at the start (we expect decrease for overfitting a fixed batch)
    assert losses[-1] < losses[0], (
        f"Loss should decrease with fix-batch: start={losses[0]:.4f} end={losses[-1]:.4f}. "
        "If this fails, gradient or LR may not be applied correctly."
    )


def test_optimizer_embedding_update_uses_lr():
    """Embedding E is updated with E_new = E_old - lr * dE."""
    cfg = TransformerConfig(_minimal_config())
    model = EinsumTransformer(cfg)
    model._init_stats()
    lr = 0.1
    optimizer = EinsumOptimizer(model, lr=lr)

    x = np.random.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    y = np.roll(x, -1, axis=1)
    y[:, -1] = 0

    logits = model.forward(x, training=True, verbose=False)
    loss, d_logits = _cross_entropy_loss(logits, y)
    model.backward(d_logits)

    E_name = model.E_name
    assert "E" in model._grads
    old_E = np.array(model.weights[E_name], copy=True)
    dE = model._grads["E"]
    optimizer.step()
    new_E = np.array(model.weights[E_name], copy=True)

    expected_E = old_E - lr * dE
    np.testing.assert_allclose(new_E, expected_E, rtol=1e-5, atol=1e-5,
        err_msg="E update must follow E_new = E_old - lr * dE")
