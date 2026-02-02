import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import core_transformer

class TestTiledGPU(unittest.TestCase):
    def test_tiled_matmul_cupy_backend(self):
        # Create dummy inputs
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20, 30).astype(np.float32)

        # Mock CuPy
        mock_cp = MagicMock()
        # Mock array creation to return a mock object
        mock_a_cp = MagicMock()
        mock_b_cp = MagicMock()
        mock_res_cp = MagicMock()

        # When cp.array is called with a, return mock_a_cp
        # When cp.array is called with b, return mock_b_cp
        def array_side_effect(x):
            if x is a: return mock_a_cp
            if x is b: return mock_b_cp
            return MagicMock()

        mock_cp.array.side_effect = array_side_effect
        mock_cp.matmul.return_value = mock_res_cp
        mock_cp.asnumpy.return_value = np.zeros((10, 30), dtype=np.float32) # Dummy result

        # We need isinstance(a, cp.ndarray) to be False for numpy inputs
        # core_transformer.cp will be this mock_cp
        # mock_cp.ndarray needs to be a class that 'a' is NOT an instance of.
        class MockNdArray: pass
        mock_cp.ndarray = MockNdArray

        # Patch core_transformer
        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', mock_cp, create=True):
                # We need to reload or ensure the function uses the patched values.
                # Since tiled_matmul uses 'cp' from global scope of core_transformer, patching core_transformer.cp works.

                res = core_transformer.tiled_matmul(a, b, backend="cupy")

                # Verify cp.array called
                self.assertEqual(mock_cp.array.call_count, 2)
                args0, _ = mock_cp.array.call_args_list[0]
                args1, _ = mock_cp.array.call_args_list[1]
                # Note order might vary? In implementation: if not is_a: a=...; if not is_b: b=...
                # So 'a' is converted first, then 'b'.
                np.testing.assert_array_equal(args0[0], a)
                np.testing.assert_array_equal(args1[0], b)

                # Verify cp.matmul called with mocked cupy arrays
                mock_cp.matmul.assert_called_once_with(mock_a_cp, mock_b_cp)

                # Verify cp.asnumpy called (since inputs were numpy)
                mock_cp.asnumpy.assert_called_once_with(mock_res_cp)

                # Verify result is numpy
                self.assertIsInstance(res, np.ndarray)

    def test_tiled_matmul_cupy_auto_detection(self):
        # Test that backend="auto" works if inputs are cupy arrays

        mock_cp = MagicMock()

        class MockNdArray: pass
        mock_cp.ndarray = MockNdArray

        mock_a_cp = MockNdArray()
        mock_b_cp = MockNdArray()
        mock_res_cp = MockNdArray()

        mock_cp.matmul.return_value = mock_res_cp

        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', mock_cp, create=True):
                res = core_transformer.tiled_matmul(mock_a_cp, mock_b_cp, backend="auto")

                # Verify matmul called
                mock_cp.matmul.assert_called_once_with(mock_a_cp, mock_b_cp)

                # Verify conversion NOT called (since inputs were cupy)
                mock_cp.asnumpy.assert_not_called()

if __name__ == '__main__':
    unittest.main()
