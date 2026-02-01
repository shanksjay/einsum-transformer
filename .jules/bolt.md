## 2024-05-23 - Einsum Performance Trap
**Learning:** `np.einsum` without `optimize=True` uses a slow C-iterator that is 10-15x slower than BLAS-dispatched `matmul` or `tensordot` for standard Transformer contractions. Adding `optimize=True` recovers this performance while maintaining the readable einsum notation.
**Action:** Always use `optimize=True` (or check if `np.matmul` applies) for performance-critical contractions in NumPy.

## 2024-05-23 - Simulated Quantization Bottleneck
**Learning:** In this "glass box" architecture, simulated quantization (dequantizing weights on-the-fly in `get_w` every step) dominates inference latency (25ms vs 2ms compute in decode).
**Action:** For future architectural changes, caching dequantized weights during inference is the highest-value optimization.

## 2026-01-30 - Optimized Sigmoid Performance
**Learning:** A naive sigmoid implementation `1.0 / (1.0 + np.exp(-x))` wrapped in `np.errstate(over='ignore')` is ~13x faster than a masked piecewise implementation and ~3x faster than `scipy.special.expit` in this environment. It avoids boolean indexing overhead and complex branching.
**Action:** Use the naive implementation with error state handling for sigmoid activation, ensuring output dtype is preserved to avoid implicit promotion to `float64`.

## 2026-10-24 - RoPE Recomputation Bottleneck
**Learning:** Contrary to documentation, RoPE embeddings were recomputed on every call, consuming ~14% of forward pass time in small-batch/training regimes.
**Action:** Implemented `_init_rope` to cache sin/cos tables. Always verify that "precomputed" features are actually stored and used, especially for geometric embeddings.

## 2026-10-25 - RMSNorm Allocation Bottleneck
**Learning:** `x.astype(np.float64)` in `rms_norm` creates a full copy of the activation tensor, doubling memory usage and bandwidth. For large tensors, this allocation is slower than the reduction itself.
**Action:** Use `np.einsum("...d,...d->...", x, x, dtype=np.float64, optimize=True)` to compute the sum of squares directly in high precision without allocating a temporary float64 input tensor. Use a size threshold (e.g., 65536) to fallback to standard implementation for small inputs where einsum overhead dominates.
