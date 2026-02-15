import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core_transformer import tiled_matmul, HAS_CUPY

@pytest.fixture
def executor():
    return ThreadPoolExecutor(max_workers=4)

def test_2d_correctness(executor):
    """Test standard 2D tiled matmul."""
    M, K, N = 128, 128, 128
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    res_tiled = tiled_matmul(a, b, executor=executor)

    np.testing.assert_allclose(res_np, res_tiled, atol=1e-5)

def test_batched_correctness_matching(executor):
    """Test batched tiled matmul with matching batch dimensions."""
    B, H, M, K = 2, 4, 32, 64
    N = 32
    # [B, H, M, K] @ [B, H, K, N] -> [B, H, M, N]
    a = np.random.randn(B, H, M, K).astype(np.float32)
    b = np.random.randn(B, H, K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    res_tiled = tiled_matmul(a, b, executor=executor)

    np.testing.assert_allclose(res_np, res_tiled, atol=1e-5)

def test_batched_correctness_broadcasting(executor):
    """Test batched tiled matmul with broadcasting (fallback to np.matmul)."""
    # [B, H, M, K] @ [K, N] (2D b, handled by original tiling logic)
    B, H, M, K = 2, 4, 32, 64
    N = 32
    a = np.random.randn(B, H, M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    res_tiled = tiled_matmul(a, b, executor=executor)
    np.testing.assert_allclose(res_np, res_tiled, atol=1e-5)

    # [B, 1, M, K] @ [1, H, K, N] (Complex broadcasting, should fallback)
    a = np.random.randn(B, 1, M, K).astype(np.float32)
    b = np.random.randn(1, H, K, N).astype(np.float32)

    res_np = np.matmul(a, b)
    res_tiled = tiled_matmul(a, b, executor=executor)
    np.testing.assert_allclose(res_np, res_tiled, atol=1e-5)

def test_out_buffer(executor):
    """Test 'out' buffer support."""
    M, K, N = 64, 64, 64
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    res = tiled_matmul(a, b, executor=executor, out=out)
    assert res is out
    np.testing.assert_allclose(res, np.matmul(a, b), atol=1e-5)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
def test_cupy_backend():
    """Test CuPy backend."""
    import cupy as cp
    M, K, N = 64, 64, 64
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    # Test 1: Explicit cupy backend with numpy inputs -> expect CuPy result (WAIT, my code logic returns res.get() if inputs were numpy!)
    # Logic: "return res if (is_a_cp or is_b_cp) else res.get()"
    # Inputs are numpy. is_a_cp is False. returns res.get().
    # So expect numpy.
    res = tiled_matmul(a_np, b_np, backend="cupy")
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, np.matmul(a_np, b_np), atol=1e-5)

    # Test 2: Explicit cupy backend with CuPy inputs -> expect CuPy result
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    res_cp = tiled_matmul(a_cp, b_cp, backend="cupy")
    assert isinstance(res_cp, cp.ndarray)
    np.testing.assert_allclose(res_cp.get(), np.matmul(a_np, b_np), atol=1e-5)

    # Test 3: Auto backend with CuPy inputs -> expect CuPy result
    res_auto = tiled_matmul(a_cp, b_cp, backend="auto")
    assert isinstance(res_auto, cp.ndarray)
    np.testing.assert_allclose(res_auto.get(), np.matmul(a_np, b_np), atol=1e-5)

    # Test 4: With out buffer (numpy)
    out_np = np.zeros((M, N), dtype=np.float32)
    res_out_np = tiled_matmul(a_cp, b_cp, backend="cupy", out=out_np)
    assert isinstance(res_out_np, np.ndarray)
    assert res_out_np is out_np
    np.testing.assert_allclose(out_np, np.matmul(a_np, b_np), atol=1e-5)

    # Test 5: With out buffer (cupy)
    out_cp = cp.empty((M, N), dtype=cp.float32)
    res_out_cp = tiled_matmul(a_cp, b_cp, backend="cupy", out=out_cp)
    assert isinstance(res_out_cp, cp.ndarray)
    assert res_out_cp is out_cp
    np.testing.assert_allclose(out_cp.get(), np.matmul(a_np, b_np), atol=1e-5)
