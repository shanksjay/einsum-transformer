
import numpy as np
import pytest
from core_transformer import tiled_matmul

def test_tiled_matmul_cpu_correctness():
    """Verify that tiled_matmul produces correct results on CPU compared to np.matmul."""
    M, K, N = 128, 64, 128
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Reference
    expected = np.matmul(a, b)

    # Tiled Matmul (default backend="auto", likely numpy on CPU)
    # block_size small to force tiling logic
    result = tiled_matmul(a, b, block_size=32, backend="numpy")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_tiled_matmul_cpu_parallel():
    """Verify tiled_matmul with executor (multi-threading)."""
    from concurrent.futures import ThreadPoolExecutor

    M, K, N = 256, 128, 256 # larger to trigger parallel threshold if heuristic used, but we force executor
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    expected = np.matmul(a, b)

    with ThreadPoolExecutor(max_workers=2) as executor:
        # We need to ensure logic uses executor.
        # tiled_matmul heuristic checks total_ops > 1.5e8.
        # 256*128*256 = 8,388,608 < 1.5e8.
        # So it won't use parallel unless we force it or lower threshold.
        # But we can't easily force it without changing code or mocking.
        # However, passing executor is enough to verify it runs without crashing.
        # To truly test parallel path, we need large matrices.

        # Let's try large matrices but maybe just check correctness.
        # M=1024, K=1024, N=1024 => 1e9 ops > 1.5e8.
        # usage: 1024*1024*4 bytes = 4MB. Fine.

        M, K, N = 1024, 1024, 1024
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        expected = np.matmul(a, b)

        result = tiled_matmul(a, b, block_size=256, executor=executor, backend="numpy")

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_tiled_matmul_out_buffer():
    """Verify 'out' parameter usage."""
    M, K, N = 64, 64, 64
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    out = np.zeros((M, N), dtype=np.float32)

    res = tiled_matmul(a, b, backend="numpy", out=out)

    assert res is out
    np.testing.assert_allclose(out, np.matmul(a, b), rtol=1e-5, atol=1e-5)
