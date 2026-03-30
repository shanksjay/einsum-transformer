import numpy as np
import pytest
from core_transformer import tiled_matmul
import sys
from unittest.mock import MagicMock

class MockCuPyArray:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim

    def get(self, out=None):
        if out is not None:
            np.copyto(out, self.data)
            return out
        return self.data.copy()

class MockCuPyModule:
    ndarray = MockCuPyArray

    @staticmethod
    def asarray(a):
        return MockCuPyArray(a)

    @staticmethod
    def matmul(a, b, out=None):
        if isinstance(a, MockCuPyArray):
            a_data = a.data
        else:
            a_data = np.asarray(a)

        if isinstance(b, MockCuPyArray):
            b_data = b.data
        else:
            b_data = np.asarray(b)

        res_data = np.matmul(a_data, b_data)

        if out is not None:
            if isinstance(out, MockCuPyArray):
                np.copyto(out.data, res_data)
                return out
            else:
                np.copyto(out, res_data)
                return out

        return MockCuPyArray(res_data)


def test_tiled_matmul_cupy_backend_no_out(monkeypatch):
    import core_transformer
    mock_cp = MockCuPyModule()

    # Inject our mock into core_transformer
    monkeypatch.setattr(core_transformer, "cp", mock_cp, raising=False)
    monkeypatch.setattr(core_transformer, "HAS_CUPY", True)

    a = np.random.rand(4, 5)
    b = np.random.rand(5, 6)

    res = core_transformer.tiled_matmul(a, b, backend="cupy")

    assert isinstance(res, MockCuPyArray)
    np.testing.assert_allclose(res.data, np.matmul(a, b))

def test_tiled_matmul_cupy_backend_with_cupy_out(monkeypatch):
    import core_transformer
    mock_cp = MockCuPyModule()
    monkeypatch.setattr(core_transformer, "cp", mock_cp, raising=False)
    monkeypatch.setattr(core_transformer, "HAS_CUPY", True)

    a = np.random.rand(4, 5)
    b = np.random.rand(5, 6)
    out = mock_cp.asarray(np.zeros((4, 6)))

    res = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)

    assert res is out
    np.testing.assert_allclose(out.data, np.matmul(a, b))

def test_tiled_matmul_cupy_backend_with_numpy_out(monkeypatch):
    import core_transformer
    mock_cp = MockCuPyModule()
    monkeypatch.setattr(core_transformer, "cp", mock_cp, raising=False)
    monkeypatch.setattr(core_transformer, "HAS_CUPY", True)

    a = np.random.rand(4, 5)
    b = np.random.rand(5, 6)
    out = np.zeros((4, 6))

    res = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)

    assert res is out
    np.testing.assert_allclose(out, np.matmul(a, b))

def test_tiled_matmul_auto_backend_with_cupy_inputs(monkeypatch):
    import core_transformer
    mock_cp = MockCuPyModule()
    monkeypatch.setattr(core_transformer, "cp", mock_cp, raising=False)
    monkeypatch.setattr(core_transformer, "HAS_CUPY", True)

    a = mock_cp.asarray(np.random.rand(4, 5))
    b = np.random.rand(5, 6) # one cupy array should trigger it

    res = core_transformer.tiled_matmul(a, b, backend="auto")

    assert isinstance(res, MockCuPyArray)
    np.testing.assert_allclose(res.data, np.matmul(a.data, b))
