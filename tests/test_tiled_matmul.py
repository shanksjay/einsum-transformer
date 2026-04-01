import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import sys
import copy

from core_transformer import tiled_matmul, HAS_CUPY

def test_tiled_matmul_2d_numpy():
    """Test standard 2D matrix multiplication with numpy backend."""
    a = np.random.randn(64, 128).astype(np.float32)
    b = np.random.randn(128, 256).astype(np.float32)

    expected = np.matmul(a, b)
    result = tiled_matmul(a, b, backend="numpy")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_tiled_matmul_batched_3d_numpy():
    """Test 3D batched matrix multiplication with numpy backend."""
    # Ensure dimensions are large enough to exceed 1.5e8 ops and trigger the parallel loop
    a = np.random.randn(4, 512, 512).astype(np.float32)
    b = np.random.randn(4, 512, 1024).astype(np.float32)

    expected = np.matmul(a, b)

    # Use ThreadPoolExecutor to trigger parallel path
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = tiled_matmul(a, b, backend="numpy", executor=executor)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_tiled_matmul_batched_4d_numpy():
    """Test 4D batched matrix multiplication with numpy backend."""
    # Ensure dimensions are large enough to exceed 1.5e8 ops and trigger the parallel loop
    a = np.random.randn(2, 2, 512, 512).astype(np.float32)
    b = np.random.randn(2, 2, 512, 1024).astype(np.float32)

    expected = np.matmul(a, b)

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = tiled_matmul(a, b, backend="numpy", executor=executor)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_tiled_matmul_out_buffer():
    """Test that out buffer is used and updated correctly."""
    a = np.random.randn(64, 128).astype(np.float32)
    b = np.random.randn(128, 256).astype(np.float32)

    out = np.zeros((64, 256), dtype=np.float32)
    expected = np.matmul(a, b)

    result = tiled_matmul(a, b, backend="numpy", out=out)

    # Check that output is correct
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    # Check that out buffer was actually modified
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    # Check that returned array shares memory with out
    assert np.shares_memory(result, out) or id(result) == id(out)

def test_tiled_matmul_batched_out_buffer():
    """Test batched matrix multiplication with out buffer."""
    # Ensure dimensions are large enough to exceed 1.5e8 ops and trigger the parallel loop
    a = np.random.randn(4, 512, 512).astype(np.float32)
    b = np.random.randn(4, 512, 1024).astype(np.float32)

    out = np.zeros((4, 512, 1024), dtype=np.float32)
    expected = np.matmul(a, b)

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = tiled_matmul(a, b, backend="numpy", executor=executor, out=out)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

def test_tiled_matmul_fallback_1d_a():
    """Test fallback when a is 1D vector."""
    a = np.random.randn(128).astype(np.float32)
    b = np.random.randn(4, 128, 256).astype(np.float32)

    expected = np.matmul(a, b)

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = tiled_matmul(a, b, backend="numpy", executor=executor)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_tiled_matmul_cupy_mock():
    """Test cupy backend with mocked cupy."""
    class MockCuPyArray:
        def __init__(self, data):
            self.data = data
        def get(self, out=None):
            if out is not None:
                out[...] = self.data
                return out
            return self.data

    mock_cp = MagicMock()
    mock_cp.ndarray = MockCuPyArray

    def asarray_mock(x):
        return MockCuPyArray(x)

    mock_cp.asarray = MagicMock(side_effect=asarray_mock)

    def matmul_mock(a, b, out=None):
        a_data = a.data if isinstance(a, MockCuPyArray) else a
        b_data = b.data if isinstance(b, MockCuPyArray) else b
        res = np.matmul(a_data, b_data)
        if out is not None:
            if isinstance(out, MockCuPyArray):
                out.data = res
            else:
                out[...] = res
            return out
        return MockCuPyArray(res)

    mock_cp.matmul = MagicMock(side_effect=matmul_mock)

    import core_transformer

    # Save original globals
    orig_has_cupy = core_transformer.HAS_CUPY
    orig_cp = getattr(core_transformer, 'cp', None)

    try:
        # Patch module globals
        core_transformer.HAS_CUPY = True
        core_transformer.cp = mock_cp

        a = np.random.randn(64, 128).astype(np.float32)
        b = np.random.randn(128, 256).astype(np.float32)
        expected = np.matmul(a, b)

        # Test basic cupy matmul
        res = core_transformer.tiled_matmul(a, b, backend="cupy")
        assert isinstance(res, MockCuPyArray)
        np.testing.assert_allclose(res.data, expected, rtol=1e-5, atol=1e-5)

        # Test cupy matmul with numpy out buffer
        out = np.zeros((64, 256), dtype=np.float32)
        res_out = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)
        assert res_out is out
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    finally:
        # Restore original globals
        core_transformer.HAS_CUPY = orig_has_cupy
        if orig_cp is not None:
            core_transformer.cp = orig_cp
        else:
            delattr(core_transformer, 'cp')
