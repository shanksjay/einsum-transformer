import pytest
import numpy as np
import sys
import os

# Ensure we can import core_transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_transformer import tiled_matmul

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def test_tiled_matmul_cpu():
    M, K, N = 128, 64, 128
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    expected = np.matmul(a, b)
    # Force block size small enough to ensure tiling loops run
    result = tiled_matmul(a, b, block_size=32)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_tiled_matmul_gpu_backend_cupy():
    M, K, N = 128, 64, 128
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Passing numpy arrays with backend="cupy" should return numpy arrays
    # (It internally converts to GPU, computes, converts back)
    result = tiled_matmul(a, b, backend="cupy")
    expected = np.matmul(a, b)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_tiled_matmul_gpu_auto():
    M, K, N = 128, 64, 128
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    # backend="auto" with cupy inputs should return cupy array
    result = tiled_matmul(a_cp, b_cp, backend="auto")

    assert isinstance(result, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(result), np.matmul(a_np, b_np), rtol=1e-5, atol=1e-5)
