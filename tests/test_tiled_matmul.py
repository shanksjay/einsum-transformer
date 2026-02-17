import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import core_transformer
from core_transformer import tiled_matmul, HAS_MLX, HAS_NUMBA
from concurrent.futures import ThreadPoolExecutor

try:
    import cupy as cp
    HAS_CUPY_INSTALLED = True
except ImportError:
    HAS_CUPY_INSTALLED = False

def test_tiled_matmul_basic():
    """Test basic 2D matrix multiplication correctness."""
    M, K, N = 128, 256, 128
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    res_tiled = tiled_matmul(a, b, block_size=64)

    np.testing.assert_allclose(res_tiled, res_np, atol=1e-4)

def test_tiled_matmul_batched():
    """Test batched matrix multiplication correctness."""
    B, M, K, N = 4, 32, 64, 32
    a = np.random.randn(B, M, K).astype(np.float32)
    b = np.random.randn(B, K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    # Pass executor to allow parallel path check (though ops might be small)
    with ThreadPoolExecutor(max_workers=2) as executor:
        res_tiled = tiled_matmul(a, b, block_size=16, executor=executor)

    np.testing.assert_allclose(res_tiled, res_np, atol=1e-4)

def test_tiled_matmul_batched_large():
    """Test batched matrix multiplication with large enough size to trigger parallelization."""
    # M*K*N must be > 1.5e8 / B_total to trigger?
    # Logic: total_ops = B_total * M * K * N
    # Let's target > 1.5e8 ops.
    # B=4. 1.5e8 / 4 = 37.5e6.
    # M=K=N=340 -> 39e6 ops.
    B, M, K, N = 2, 350, 350, 350
    # Use float32 to save memory, total size ~350^3 * 2 * 4 bytes ~ 340MB? No.
    # 2 * 350*350 floats * 4 bytes = 2 * 122500 * 4 = 1MB per matrix.
    # Ops count is what matters. 2 * 350^3 = 85 million. Still < 1.5e8.
    # Need larger. B=4, M=K=N=350 -> 170 million.
    B = 4

    a = np.random.randn(B, M, K).astype(np.float32)
    b = np.random.randn(B, K, N).astype(np.float32)

    res_np = np.matmul(a, b)

    with ThreadPoolExecutor(max_workers=4) as executor:
        res_tiled = tiled_matmul(a, b, executor=executor) # block_size default

    np.testing.assert_allclose(res_tiled, res_np, atol=1e-4)

def test_tiled_matmul_broadcast():
    """Test batched matrix multiplication with broadcasting."""
    B, M, K, N = 4, 32, 64, 32
    # a: [B, M, K], b: [K, N] -> Broadcast b over B
    a = np.random.randn(B, M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    # This falls back to 2D broadcast path, not batched path logic
    res_tiled = tiled_matmul(a, b, block_size=16)

    np.testing.assert_allclose(res_tiled, res_np, atol=1e-4)

def test_tiled_matmul_batched_broadcast_dims():
    """Test batched matrix multiplication where broadcasting happens on batch dims."""
    # a: [B, 1, M, K], b: [1, H, K, N] -> [B, H, M, N]
    B, H, M, K, N = 2, 3, 32, 32, 32
    a = np.random.randn(B, 1, M, K).astype(np.float32)
    b = np.random.randn(1, H, K, N).astype(np.float32)

    res_np = np.matmul(a, b)

    # This logic handles complex broadcasting via np.matmul fallback or manual broadcasting?
    # My implementation checks `if a.shape[:-2] == b.shape[:-2]`.
    # So [B, 1] != [1, H]. It should fall back to np.matmul.
    res_tiled = tiled_matmul(a, b)

    np.testing.assert_allclose(res_tiled, res_np, atol=1e-4)


@pytest.mark.skipif(not HAS_CUPY_INSTALLED, reason="CuPy not installed")
def test_tiled_matmul_cupy_real():
    """Test Real CuPy backend."""
    M, K, N = 128, 256, 128
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    try:
        res_cupy = tiled_matmul(a_np, b_np, backend="cupy")
        assert isinstance(res_cupy, cp.ndarray)
        res_np = np.matmul(a_np, b_np)
        np.testing.assert_allclose(cp.asnumpy(res_cupy), res_np, atol=1e-4)
    except Exception as e:
        pytest.fail(f"CuPy execution failed: {e}")

def test_tiled_matmul_cupy_mock():
    """Test CuPy path using mocks when CuPy is not installed."""
    if core_transformer.HAS_CUPY:
        # If installed, we don't mock, we trust the real test
        return

    # Mock cp
    mock_cp = MagicMock()
    class MockArray:
        pass
    mock_cp.ndarray = MockArray
    mock_cp.array.side_effect = lambda x: MockArray()
    mock_cp.matmul.return_value = MockArray()
    mock_cp.asnumpy.return_value = np.zeros((10, 10))

    # Patch modules in core_transformer
    with patch('core_transformer.HAS_CUPY', True), \
         patch('core_transformer.cp', mock_cp, create=True):

        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)

        # Test backend="cupy"
        res = tiled_matmul(a, b, backend="cupy")

        assert mock_cp.array.called # Should convert inputs
        assert mock_cp.matmul.called
        assert isinstance(res, MockArray)

        # Test output copy back
        out = np.zeros((10, 10))
        res = tiled_matmul(a, b, backend="cupy", out=out)
        assert mock_cp.asnumpy.called
        assert res is out

if __name__ == "__main__":
    # Manual run
    print("Running basic test...")
    test_tiled_matmul_basic()
    print("Running batched test...")
    test_tiled_matmul_batched()
    print("Running large batched test...")
    test_tiled_matmul_batched_large()
    print("Running complex broadcast test...")
    test_tiled_matmul_batched_broadcast_dims()
    if not HAS_CUPY_INSTALLED:
        print("Running CuPy mock test...")
        test_tiled_matmul_cupy_mock()
    else:
        print("Running CuPy real test...")
        test_tiled_matmul_cupy_real()
    print("All tests passed!")
