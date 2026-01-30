## 2024-05-23 - Einsum Performance Trap
**Learning:** `np.einsum` without `optimize=True` uses a slow C-iterator that is 10-15x slower than BLAS-dispatched `matmul` or `tensordot` for standard Transformer contractions. Adding `optimize=True` recovers this performance while maintaining the readable einsum notation.
**Action:** Always use `optimize=True` (or check if `np.matmul` applies) for performance-critical contractions in NumPy.

## 2024-05-23 - Simulated Quantization Bottleneck
**Learning:** In this "glass box" architecture, simulated quantization (dequantizing weights on-the-fly in `get_w` every step) dominates inference latency (25ms vs 2ms compute in decode).
**Action:** For future architectural changes, caching dequantized weights during inference is the highest-value optimization.

## 2026-01-30 - Optimized Sigmoid Performance
**Learning:** A naive sigmoid implementation `1.0 / (1.0 + np.exp(-x))` wrapped in `np.errstate(over='ignore')` is ~13x faster than a masked piecewise implementation and ~3x faster than `scipy.special.expit` in this environment. It avoids boolean indexing overhead and complex branching.
**Action:** Use the naive implementation with error state handling for sigmoid activation, ensuring output dtype is preserved to avoid implicit promotion to `float64`.
