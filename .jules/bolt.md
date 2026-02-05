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

## 2026-10-27 - ThreadPoolExecutor Overhead on Small/Medium Matrices
**Learning:** Using `ThreadPoolExecutor` for matrix multiplications with less than ~150M FLOPs or small batch sizes (M < 32) can be significantly slower (up to 30x) than single-threaded BLAS (NumPy) due to Python threading overhead and contention.
**Action:** Increased `tiled_matmul` parallelization threshold to 1.5e8 ops and disabled it for M < 32. Implemented buffer reuse for inference to further reduce allocation overhead.

## 2026-10-28 - Threading Overhead in SwiGLU and Attention
**Learning:** Unconditional submission of `swiglu` and `attention` projections to `ThreadPoolExecutor` causes significant overhead (up to ~30-40%) for small operations (e.g., L=1 decoding or small batches), where the cost of task scheduling exceeds the parallelization gain.
**Action:** Implemented `_should_parallelize(ops)` helper with a threshold of 1.5e8 ops (consistent with `tiled_matmul`) to enforce serial execution for small workloads in `swiglu` and `attention`.

## 2024-10-31 - Micro-benchmark Denormal Trap
**Learning:** Micro-benchmarking SwiGLU activation with repeated loops on the same data caused denormals/underflow, making optimized in-place code 100x slower. In-place operations on buffers that decay to zero trigger slow CPU denormal handling.
**Action:** Always verify micro-benchmarks with fresh random data or ensure values stay in normal range (e.g. reset periodically).
