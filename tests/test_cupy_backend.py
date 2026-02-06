
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# We need to mock cupy before importing core_transformer if we want to test the import logic,
# but since we want to test the *logic inside tiled_matmul*, we can just patch the module attributes.

import core_transformer

class TestTiledMatmulCupy(unittest.TestCase):
    def setUp(self):
        self.mock_cp = MagicMock()
        self.mock_cp.ndarray = MagicMock()
        # Make isinstance(x, mock_cp.ndarray) work
        self.mock_cp.ndarray.__class__ = type

        # Setup mock array
        self.mock_array = MagicMock()
        # To pass isinstance checks against mock_cp.ndarray, we can't easily do it if ndarray is a Mock.
        # But core_transformer checks `isinstance(a, cp.ndarray)`.
        # So we need cp.ndarray to be a type.

        class MockNdArray:
            pass
        self.mock_cp.ndarray = MockNdArray

        self.mock_cp.array.return_value = MockNdArray()
        self.mock_cp.matmul.return_value = MockNdArray()

    def test_cupy_backend_explicit(self):
        # Patch core_transformer globals
        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', self.mock_cp, create=True):
                a = np.random.randn(10, 10)
                b = np.random.randn(10, 10)

                # Call with backend="cupy"
                res = core_transformer.tiled_matmul(a, b, backend="cupy")

                # Verify cp.array called (conversion)
                self.assertEqual(self.mock_cp.array.call_count, 2)
                # Verify cp.matmul called
                self.mock_cp.matmul.assert_called()
                self.assertIsInstance(res, self.mock_cp.ndarray)

    def test_cupy_backend_auto_with_cupy_input(self):
        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', self.mock_cp, create=True):
                a = self.mock_cp.ndarray()
                b = self.mock_cp.ndarray()

                # Call with backend="auto"
                res = core_transformer.tiled_matmul(a, b, backend="auto")

                # Verify cp.matmul called directly
                self.mock_cp.matmul.assert_called_with(a, b)
                self.assertIsInstance(res, self.mock_cp.ndarray)

    def test_cupy_backend_auto_with_numpy_input(self):
        # Should fallback to CPU (tiled) if inputs are numpy and backend="auto",
        # UNLESS we enforce "auto" means "use GPU if available"?
        # Current logic:
        # if (backend == "auto" and HAS_CUPY) and (isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray)): ...
        # It requires at least one input to be cupy array to trigger auto-cupy path.
        # This matches standard behavior (dispatch on input type).

        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', self.mock_cp, create=True):
                a = np.random.randn(10, 10)
                b = np.random.randn(10, 10)

                # Call with backend="auto"
                # Should NOT call cp.matmul because inputs are numpy
                res = core_transformer.tiled_matmul(a, b, backend="auto")

                self.mock_cp.matmul.assert_not_called()
                self.assertTrue(isinstance(res, np.ndarray))

if __name__ == "__main__":
    unittest.main()
