import numpy as np
import pytest
import sys
from unittest.mock import MagicMock

# Create a mock cupy module
mock_cp = MagicMock()
mock_cp.array = MagicMock(side_effect=lambda x: MockCupyArray(x))

def mock_matmul(a, b, out=None):
    a_np = a.data if isinstance(a, MockCupyArray) else a
    b_np = b.data if isinstance(b, MockCupyArray) else b

    res_np = np.matmul(a_np, b_np)

    if out is not None:
        if hasattr(out, 'get'):
            np.copyto(out.data, res_np)
            return out
        else:
            np.copyto(out, res_np)
            return out
    else:
        return MockCupyArray(res_np)

mock_cp.matmul = MagicMock(side_effect=mock_matmul)

class MockCupyArray:
    def __init__(self, data):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data

    def get(self, out=None):
        if out is not None:
            np.copyto(out, self.data)
            return out
        return self.data.copy()

# Manually inject our mock cupy into sys.modules BEFORE importing core_transformer
sys.modules['cupy'] = mock_cp

import core_transformer

def setup_module():
    # Make sure core_transformer sees the mock
    core_transformer.cp = mock_cp
    core_transformer.HAS_CUPY = True

def test_tiled_matmul_cupy_backend_auto():
    a_np = np.random.randn(10, 10)
    b_np = np.random.randn(10, 10)

    a_cp = MockCupyArray(a_np)
    b_cp = MockCupyArray(b_np)

    # Auto backend with cupy arrays
    mock_cp.matmul.reset_mock()
    res = core_transformer.tiled_matmul(a_cp, b_cp, backend="auto")
    assert isinstance(res, MockCupyArray)
    mock_cp.matmul.assert_called_once()

def test_tiled_matmul_cupy_backend_explicit():
    a_np = np.random.randn(10, 10)
    b_np = np.random.randn(10, 10)

    # Force cupy backend with numpy arrays
    mock_cp.array.reset_mock()
    mock_cp.matmul.reset_mock()
    res = core_transformer.tiled_matmul(a_np, b_np, backend="cupy")
    assert isinstance(res, MockCupyArray)
    # cp.array should be called twice to convert a_np and b_np
    assert mock_cp.array.call_count == 2
    mock_cp.matmul.assert_called_once()

def test_tiled_matmul_cupy_out_numpy():
    a_np = np.random.randn(10, 10)
    b_np = np.random.randn(10, 10)
    out_np = np.zeros((10, 10))

    a_cp = MockCupyArray(a_np)

    mock_cp.matmul.reset_mock()
    res = core_transformer.tiled_matmul(a_cp, b_np, backend="auto", out=out_np)

    # The result should be the numpy array updated in-place
    assert res is out_np
    # Verify the contents match the expected matmul
    expected = np.matmul(a_np, b_np)
    np.testing.assert_allclose(out_np, expected)

def test_tiled_matmul_cupy_out_cupy():
    a_np = np.random.randn(10, 10)
    b_np = np.random.randn(10, 10)
    out_np = np.zeros((10, 10))

    a_cp = MockCupyArray(a_np)
    out_cp = MockCupyArray(out_np)

    mock_cp.matmul.reset_mock()
    res = core_transformer.tiled_matmul(a_cp, b_np, backend="auto", out=out_cp)

    # The result should be the cupy array
    assert res is out_cp
    # Verify the contents match the expected matmul
    expected = np.matmul(a_np, b_np)
    np.testing.assert_allclose(out_cp.data, expected)
