import numpy as np
import pytest
from unittest.mock import MagicMock
import core_transformer
from core_transformer import tiled_matmul

# Helper to verify numpy implementation
def test_tiled_matmul_numpy_correctness():
    """Verify tiled_matmul produces correct results with NumPy backend."""
    # Test case 1: Small matrices (fallback to np.matmul)
    a = np.random.randn(10, 20).astype(np.float32)
    b = np.random.randn(20, 30).astype(np.float32)
    res = tiled_matmul(a, b, backend="numpy")
    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

    # Test case 2: Larger matrices to trigger tiling (force block size)
    # We force block_size=4 to ensure tiling loops run even on small data
    a = np.random.randn(32, 64).astype(np.float32)
    b = np.random.randn(64, 32).astype(np.float32)
    res = tiled_matmul(a, b, block_size=16, backend="numpy")
    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

    # Test case 3: Batched input [B, T, K] @ [K, N]
    a = np.random.randn(2, 10, 20).astype(np.float32)
    b = np.random.randn(20, 30).astype(np.float32)
    res = tiled_matmul(a, b, backend="numpy")
    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

def test_tiled_matmul_cupy_backend():
    """Verify tiled_matmul dispatches to Cupy when backend='cupy'."""
    # Mock Cupy
    mock_cp = MagicMock()
    mock_array_cls = type("MockCupyArray", (), {}) # Not inheriting from np.ndarray
    mock_cp.ndarray = mock_array_cls

    # Setup mock array instance
    mock_arr_instance = mock_array_cls()
    mock_cp.array.return_value = mock_arr_instance
    mock_cp.matmul.return_value = mock_arr_instance

    # Monkeypatch core_transformer
    original_cp = getattr(core_transformer, "cp", None)
    original_has_cupy = core_transformer.HAS_CUPY

    core_transformer.cp = mock_cp
    core_transformer.HAS_CUPY = True

    try:
        a = np.random.randn(10, 20)
        b = np.random.randn(20, 30)

        # 1. Force backend="cupy"
        res = tiled_matmul(a, b, backend="cupy")

        # Verify inputs converted to cupy
        assert mock_cp.array.call_count >= 2
        # Verify matmul called
        mock_cp.matmul.assert_called()
        # Verify result is the mock array
        assert res is mock_arr_instance

        # 2. Reset and test backend="auto" with cupy inputs
        mock_cp.reset_mock()

        # Create mock cupy inputs
        a_cp = mock_array_cls()
        b_cp = mock_array_cls()

        res = tiled_matmul(a_cp, b_cp, backend="auto")

        # Verify matmul called directly
        mock_cp.matmul.assert_called_with(a_cp, b_cp, out=None)
        # Verify no conversion (array() not called)
        mock_cp.array.assert_not_called()

    finally:
        # Restore original state
        if original_cp:
            core_transformer.cp = original_cp
        else:
            del core_transformer.cp
        core_transformer.HAS_CUPY = original_has_cupy
