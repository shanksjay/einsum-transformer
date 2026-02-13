import numpy as np
import pytest
from core_transformer import tiled_matmul

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def test_tiled_matmul_2d_numpy():
    M, K, N = 128, 64, 128
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    res = tiled_matmul(a, b, block_size=32, backend="numpy")
    expected = np.matmul(a, b)

    np.testing.assert_allclose(res, expected, atol=1e-5)

def test_tiled_matmul_batched_numpy():
    B, M, K, N = 4, 32, 64, 32
    a = np.random.randn(B, M, K).astype(np.float32)
    b = np.random.randn(B, K, N).astype(np.float32)

    # tiled_matmul should handle batch dims automatically now
    res = tiled_matmul(a, b, block_size=16, backend="numpy")
    expected = np.matmul(a, b)

    np.testing.assert_allclose(res, expected, atol=1e-5)

def test_tiled_matmul_broadcast_batch():
    # a: [B, M, K], b: [K, N] -> [B, M, N]
    B, M, K, N = 4, 32, 64, 32
    a = np.random.randn(B, M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    res = tiled_matmul(a, b, block_size=16, backend="numpy")
    expected = np.matmul(a, b)

    np.testing.assert_allclose(res, expected, atol=1e-5)

def test_tiled_matmul_out_buffer():
    M, K, N = 64, 64, 64
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    res = tiled_matmul(a, b, block_size=32, backend="numpy", out=out)
    assert res is out
    np.testing.assert_allclose(res, np.matmul(a, b), atol=1e-5)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_tiled_matmul_cupy():
    M, K, N = 128, 64, 128
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    # Test with backend="cupy" passing numpy arrays (auto transfer)
    res = tiled_matmul(a_np, b_np, backend="cupy")
    expected = np.matmul(a_np, b_np)

    assert isinstance(res, np.ndarray) # Should return numpy array if inputs were numpy
    np.testing.assert_allclose(res, expected, atol=1e-5)

    # Test with cupy arrays
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)
    res_cp = tiled_matmul(a_cp, b_cp, backend="cupy")

    # backend="cupy" returns numpy as per implementation
    assert isinstance(res_cp, np.ndarray)
    np.testing.assert_allclose(res_cp, expected, atol=1e-5)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_tiled_matmul_auto_cupy():
    M, K, N = 128, 64, 128
    a_cp = cp.random.randn(M, K).astype(np.float32)
    b_cp = cp.random.randn(K, N).astype(np.float32)

    # backend="auto" should detect cupy inputs and stay on GPU
    res = tiled_matmul(a_cp, b_cp, backend="auto")

    assert isinstance(res, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(res), cp.asnumpy(a_cp @ b_cp), atol=1e-5)
