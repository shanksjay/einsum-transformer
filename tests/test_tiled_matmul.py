import numpy as np
import pytest
import unittest.mock
import core_transformer

# Mock cupy array that DOES NOT inherit from np.ndarray
class MockCupyArray:
    def __init__(self, data):
        self.data = np.array(data)

    def get(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"MockCupyArray({self.data})"

    # Support simple slicing/indexing if needed by tiled_matmul?
    # tiled_matmul only calls cp.matmul(a, b). It doesn't slice cupy arrays inside use_cupy block.
    # But checking source:
    # if use_cupy: ... res = cp.matmul(a, b) ... return res
    # It does not do slicing on `a` or `b` inside `if use_cupy`.
    # It assumes cp.matmul handles it.

    # However, tiled_matmul logic checks b.ndim > 2 before.
    # But that's before usage.

# Prepare mock cupy module
mock_cp = unittest.mock.MagicMock()
mock_cp.ndarray = MockCupyArray
mock_cp.array = lambda x: MockCupyArray(x)

def matmul_impl(a, b):
    data_a = a.data if isinstance(a, MockCupyArray) else a
    data_b = b.data if isinstance(b, MockCupyArray) else b
    res = np.matmul(data_a, data_b)
    return MockCupyArray(res)

mock_cp.matmul = matmul_impl
mock_cp.copyto = lambda dst, src: dst.data[:] == src.data[:] if isinstance(dst, MockCupyArray) else None
# Note: real cp.copyto handles various cases, here we mock what we need.
# But core_transformer uses cp.copyto(out, res) where out is cp.ndarray.
# If out is np.ndarray, it uses out[:] = res.get().

def test_tiled_matmul_cpu():
    M, K, N = 10, 10, 10
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Should use CPU implementation (numpy backend or auto fallback)
    res = core_transformer.tiled_matmul(a, b, backend="numpy")
    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected, atol=1e-5)

def test_tiled_matmul_cupy_explicit():
    # Patch HAS_CUPY and cp
    with unittest.mock.patch('core_transformer.HAS_CUPY', True), \
         unittest.mock.patch('core_transformer.cp', mock_cp, create=True):

        M, K, N = 10, 10, 10
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        res = core_transformer.tiled_matmul(a, b, backend="cupy")

        # It should return a MockCupyArray (simulating cupy array)
        assert isinstance(res, MockCupyArray)

        expected = np.matmul(a, b)
        np.testing.assert_allclose(res.get(), expected, atol=1e-5)

def test_tiled_matmul_auto_numpy():
    # Test auto backend with numpy input -> transparent transfer
    with unittest.mock.patch('core_transformer.HAS_CUPY', True), \
         unittest.mock.patch('core_transformer.cp', mock_cp, create=True):

        M, K, N = 10, 10, 10
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        # backend="auto" with numpy inputs -> transparent transfer -> return numpy
        res = core_transformer.tiled_matmul(a, b, backend="auto")

        # Should return standard numpy array (via .get())
        assert isinstance(res, np.ndarray)
        assert not isinstance(res, MockCupyArray)

        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, atol=1e-5)

def test_tiled_matmul_auto_cupy():
    # Test auto backend with cupy input -> stay on GPU
    with unittest.mock.patch('core_transformer.HAS_CUPY', True), \
         unittest.mock.patch('core_transformer.cp', mock_cp, create=True):

        M, K, N = 10, 10, 10
        # Simulate inputs already on GPU
        a = MockCupyArray(np.random.randn(M, K).astype(np.float32))
        b = MockCupyArray(np.random.randn(K, N).astype(np.float32))

        # backend="auto" with cupy inputs -> return cupy
        res = core_transformer.tiled_matmul(a, b, backend="auto")

        assert isinstance(res, MockCupyArray)

        expected = np.matmul(a.get(), b.get())
        np.testing.assert_allclose(res.get(), expected, atol=1e-5)

def test_tiled_matmul_cupy_out_numpy():
    # Test cupy backend with numpy output buffer
    with unittest.mock.patch('core_transformer.HAS_CUPY', True), \
         unittest.mock.patch('core_transformer.cp', mock_cp, create=True):

        M, K, N = 10, 10, 10
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        res = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)

        # Result should be returned in out, and be numpy array
        assert res is out
        assert isinstance(res, np.ndarray)

        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, atol=1e-5)
