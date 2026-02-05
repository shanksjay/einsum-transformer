import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import core_transformer
sys.path.append(os.getcwd())
import core_transformer
from core_transformer import tiled_matmul

class TestTiledMatmul(unittest.TestCase):
    def setUp(self):
        self.a = np.random.randn(32, 64).astype(np.float32)
        self.b = np.random.randn(64, 32).astype(np.float32)
        self.expected = np.matmul(self.a, self.b)

    def test_cpu_serial(self):
        res = tiled_matmul(self.a, self.b, block_size=16, executor=None)
        np.testing.assert_allclose(res, self.expected, rtol=1e-4, atol=1e-4)

    def test_cupy_mocked(self):
        # Force HAS_CUPY to True
        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', create=True) as mock_cp:
                # Mock ndarray class
                class MockNdArray: pass
                mock_cp.ndarray = MockNdArray

                # Mock array() to return a MockNdArray instance
                def side_effect_array(x):
                    return MockNdArray()
                mock_cp.array.side_effect = side_effect_array

                # Mock matmul
                mock_cp.matmul.return_value = "cupy_result"

                # Case 1: backend="cupy"
                res = tiled_matmul(self.a, self.b, backend="cupy")
                self.assertEqual(res, "cupy_result")

                # Verify cp.array was called twice (for a and b conversion)
                self.assertEqual(mock_cp.array.call_count, 2)

    def test_cupy_auto_mocked(self):
         with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', create=True) as mock_cp:
                class MockNdArray: pass
                mock_cp.ndarray = MockNdArray
                mock_cp.matmul.return_value = "cupy_result"

                # Create fake cupy arrays
                a_cp = MockNdArray()
                b_cp = MockNdArray()

                # Case 2: backend="auto" with cp inputs
                res = tiled_matmul(a_cp, b_cp, backend="auto")
                self.assertEqual(res, "cupy_result")

                # Verify cp.array was NOT called (inputs already cp)
                mock_cp.array.assert_not_called()

if __name__ == '__main__':
    unittest.main()
