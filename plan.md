1. **Understand current implementation:** The current `tiled_matmul` function in `core_transformer.py` (lines 50-156) uses `np.matmul` by default or tiling via multi-threading over 2D block shapes for large `a` matrices if `b` is exactly 2D. However, if `b.ndim > 2`, it falls back to native single-threaded `np.matmul(a, b, out=out)`. Also, there is a placeholder for `cupy` backend support but it's not implemented (only mlx and numba are present).

2. **Enhance `tiled_matmul` in `core_transformer.py`:**
   - Add CuPy support: If `backend == "cupy"` or `backend == "auto"`, try importing `cupy` and computing the matmul on GPU, correctly handling the `out` parameter and array conversion.
   - Add multi-threading support for batched inputs: When `b.ndim > 2` and operations exceed the heuristic threshold, flatten the batch dimensions, broadcast `a` properly (or just reshape `a` and `b`), and parallelize across batches using `executor.submit`.
   - Keep the original 2D tiling logic for unbatched `b.ndim == 2`.

3. **Verify functionality:**
   - Check against original results with synthetic data.
   - Run benchmark with multi-threading to see impact.
   - Run `pytest tests/`.

4. **Complete pre-commit steps:**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

5. **Submit:** Submit changes.
