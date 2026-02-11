import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Import core_transformer once at module level to avoid PyO3 reload issues
# This also ensures safetensors is loaded once.
try:
    import core_transformer
except ImportError:
    # If path setup is needed (should be handled by uv run/env but just in case)
    sys.path.append('.')
    import core_transformer

# Define a mock class for cupy.ndarray that does NOT inherit from np.ndarray
class MockNdarray:
    def __init__(self, array):
        self.array = array

    def __getattr__(self, name):
        return getattr(self.array, name)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __repr__(self):
        return f"MockNdarray({self.array})"

# Create a mock cupy module
mock_cp = MagicMock()
mock_cp.ndarray = MockNdarray

def asarray_side_effect(x):
    if isinstance(x, MockNdarray): return x
    return MockNdarray(x)
mock_cp.asarray = MagicMock(side_effect=asarray_side_effect)

def asnumpy_side_effect(x):
    if isinstance(x, MockNdarray): return x.array
    return x
mock_cp.asnumpy = MagicMock(side_effect=asnumpy_side_effect)

def matmul_side_effect(a, b):
    a_arr = a.array if isinstance(a, MockNdarray) else a
    b_arr = b.array if isinstance(b, MockNdarray) else b
    try:
        res = np.matmul(a_arr, b_arr)
    except Exception as e:
        print(f"Matmul failed: {e}")
        raise
    return MockNdarray(res)
mock_cp.matmul = MagicMock(side_effect=matmul_side_effect)

def copyto_side_effect(dst, src):
    dst_arr = dst.array if isinstance(dst, MockNdarray) else dst
    src_arr = src.array if isinstance(src, MockNdarray) else src
    np.copyto(dst_arr, src_arr)
mock_cp.copyto = MagicMock(side_effect=copyto_side_effect)

class TestTiledMatmulGPU(unittest.TestCase):
    def setUp(self):
        # Save original state
        self.original_has_cupy = getattr(core_transformer, 'HAS_CUPY', False)
        self.original_cp = getattr(core_transformer, 'cp', None)

        # Inject mock
        core_transformer.HAS_CUPY = True
        core_transformer.cp = mock_cp

        # We also need to inject cupy into sys.modules because core_transformer.tiled_matmul
        # might check isinstance(a, cp.ndarray) where cp comes from core_transformer.cp
        # BUT if we assign core_transformer.cp = mock_cp, then cp.ndarray is mock_cp.ndarray.
        # So we don't strictly need cupy in sys.modules unless code does `import cupy`.
        # core_transformer imports it at top level.

    def tearDown(self):
        # Restore original state
        core_transformer.HAS_CUPY = self.original_has_cupy
        if self.original_cp:
            core_transformer.cp = self.original_cp
        else:
            if hasattr(core_transformer, 'cp'):
                del core_transformer.cp

    def test_tiled_matmul_cupy_backend_numpy_inputs(self):
        """Test backend='cupy' with NumPy inputs triggers transfer -> compute -> transfer."""
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20, 30).astype(np.float32)

        mock_cp.asarray.reset_mock()
        mock_cp.asnumpy.reset_mock()
        mock_cp.matmul.reset_mock()

        res = core_transformer.tiled_matmul(a, b, backend="cupy")

        # Verify calls
        self.assertTrue(mock_cp.asarray.called, "cp.asarray should be called to transfer inputs")
        self.assertTrue(mock_cp.matmul.called, "cp.matmul should be called")
        self.assertTrue(mock_cp.asnumpy.called, "cp.asnumpy should be called to transfer result back")

        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    def test_tiled_matmul_cupy_backend_batched(self):
        """Test batched matmul with backend='cupy'."""
        # a: [2, 10, 20], b: [20, 30]
        a = np.random.randn(2, 10, 20).astype(np.float32)
        b = np.random.randn(20, 30).astype(np.float32)

        res = core_transformer.tiled_matmul(a, b, backend="cupy")

        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    def test_tiled_matmul_cupy_backend_out_numpy(self):
        """Test backend='cupy' with 'out' parameter as NumPy array."""
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20, 30).astype(np.float32)
        out = np.zeros((10, 30), dtype=np.float32)

        mock_cp.asnumpy.reset_mock()

        res = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)

        expected = np.matmul(a, b)
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        np.testing.assert_allclose(res, expected, rtol=1e-5)
        self.assertTrue(mock_cp.asnumpy.called)

    def test_auto_detection_cupy(self):
        """Test backend='auto' detects CuPy inputs."""
        a_real = np.random.randn(10, 20).astype(np.float32)
        b_real = np.random.randn(20, 30).astype(np.float32)

        a = MockNdarray(a_real)
        b = MockNdarray(b_real)

        mock_cp.matmul.reset_mock()
        mock_cp.asarray.reset_mock()

        res = core_transformer.tiled_matmul(a, b, backend="auto")

        self.assertTrue(mock_cp.matmul.called)

        # Check that asarray was NOT called (logic: is_numpy_a is False)
        mock_cp.asarray.assert_not_called()

        # Result should be MockNdarray
        self.assertIsInstance(res, MockNdarray)
        np.testing.assert_allclose(res.array, np.matmul(a_real, b_real), rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
