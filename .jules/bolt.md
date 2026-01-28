## 2024-05-23 - Einsum Performance Trap
**Learning:** `np.einsum` without `optimize=True` uses a slow C-iterator that is 10-15x slower than BLAS-dispatched `matmul` or `tensordot` for standard Transformer contractions. Adding `optimize=True` recovers this performance while maintaining the readable einsum notation.
**Action:** Always use `optimize=True` (or check if `np.matmul` applies) for performance-critical contractions in NumPy.

## 2024-05-23 - Simulated Quantization Bottleneck
**Learning:** In this "glass box" architecture, simulated quantization (dequantizing weights on-the-fly in `get_w` every step) dominates inference latency (25ms vs 2ms compute in decode).
**Action:** For future architectural changes, caching dequantized weights during inference is the highest-value optimization.

## 2024-05-24 - Vectorized Sigmoid Optimization
**Learning:** Boolean masking in NumPy (`x[mask]`) incurs significant overhead from indexing and allocation. Replacing a piecewise masked sigmoid with a vectorized version (using `np.errstate` to handle overflow) yielded a ~13x speedup for the activation function.
**Action:** Prefer fully vectorized operations with `np.errstate` over boolean masking for element-wise functions when safe.
