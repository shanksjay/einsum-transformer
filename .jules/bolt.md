## 2024-05-23 - Einsum Performance Trap
**Learning:** `np.einsum` without `optimize=True` uses a slow C-iterator that is 10-15x slower than BLAS-dispatched `matmul` or `tensordot` for standard Transformer contractions. Adding `optimize=True` recovers this performance while maintaining the readable einsum notation.
**Action:** Always use `optimize=True` (or check if `np.matmul` applies) for performance-critical contractions in NumPy.

## 2024-05-23 - Simulated Quantization Bottleneck
**Learning:** In this "glass box" architecture, simulated quantization (dequantizing weights on-the-fly in `get_w` every step) dominates inference latency (25ms vs 2ms compute in decode).
**Action:** For future architectural changes, caching dequantized weights during inference is the highest-value optimization.

## 2026-01-27 - NumPy Masking Overhead
**Learning:** Piecewise function application using boolean masks in NumPy (`mask = x > 0; out[mask] = ...`) is significantly slower (10x+) than vectorized operations, even when handling special cases like overflow. Suppressing warnings with `np.errstate` is much cheaper than masking.
**Action:** Replace masked piecewise logic with mathematically equivalent continuous functions where possible, using `np.errstate` to handle safe overflows (e.g. `exp(-large) -> inf`).
