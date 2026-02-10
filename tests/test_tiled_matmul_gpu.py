import unittest
from unittest.mock import MagicMock
import sys
import numpy as np

# create mock cupy module
mock_cp = MagicMock()
class MockNdArray:
    pass
mock_cp.ndarray = MockNdArray

# Inject into sys.modules
sys.modules["cupy"] = mock_cp

# Ensure core_transformer is imported after mocking
if "core_transformer" in sys.modules:
    del sys.modules["core_transformer"]
from core_transformer import tiled_matmul

class TestTiledMatmulGPU(unittest.TestCase):
    def setUp(self):
        mock_cp.reset_mock()
        # Default behaviors
        mock_cp.asarray.side_effect = lambda x: x
        mock_cp.asnumpy.side_effect = lambda x: x
        mock_cp.matmul.return_value = np.zeros((10, 10))

    def test_backend_cupy_conversion(self):
        """Test that numpy inputs are converted to cupy when backend='cupy'."""
        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)

        tiled_matmul(a, b, backend="cupy")

        # Should call asarray to convert to GPU
        self.assertTrue(mock_cp.asarray.called)
        # Should call matmul
        self.assertTrue(mock_cp.matmul.called)
        # Should call asnumpy to convert back
        self.assertTrue(mock_cp.asnumpy.called)

    def test_backend_auto_gpu_input(self):
        """Test that passing 'gpu' arrays (mock objects) triggers cupy path with backend='auto'."""
        a_gpu = MockNdArray()
        b_gpu = MockNdArray()

        # Setup return for matmul to be a MockNdArray (simulate GPU result)
        mock_cp.matmul.return_value = MockNdArray()

        res = tiled_matmul(a_gpu, b_gpu, backend="auto")

        # Should NOT call asarray because inputs are already MockNdArray
        mock_cp.asarray.assert_not_called()

        # Should call matmul
        mock_cp.matmul.assert_called_with(a_gpu, b_gpu)

        # Should NOT call asnumpy because inputs were GPU -> return GPU
        mock_cp.asnumpy.assert_not_called()

        self.assertIsInstance(res, MockNdArray)

    def test_out_parameter_copy(self):
        """Test that out parameter is respected when converting back to numpy."""
        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)
        out = np.zeros((10, 10))

        expected = np.ones((10, 10))
        # Clear side_effect so return_value is used
        mock_cp.asnumpy.side_effect = None
        mock_cp.asnumpy.return_value = expected
        mock_cp.matmul.return_value = MockNdArray()

        res = tiled_matmul(a, b, backend="cupy", out=out)

        # Should have copied expected to out
        np.testing.assert_array_equal(out, expected)
        # Should return out
        self.assertIs(res, out)

if __name__ == "__main__":
    unittest.main()
