import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Ensure we can import core_transformer
sys.path.append(os.getcwd())
# Re-import to ensure we get fresh state if needed, though mostly we rely on patching
import core_transformer
from core_transformer import tiled_matmul, HAS_CUPY

class TestTiledMatmul(unittest.TestCase):
    def test_tiled_matmul_cpu(self):
        M, K, N = 128, 128, 128
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        # Test default (numpy)
        res = tiled_matmul(a, b, backend="numpy")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, atol=1e-5)

        # Test auto (numpy fallback)
        res_auto = tiled_matmul(a, b, backend="auto")
        np.testing.assert_allclose(res_auto, expected, atol=1e-5)

    def test_tiled_matmul_gpu_mock(self):
        # Mock cupy to test the logic path even if cupy is not installed
        # We modify global variables in core_transformer to simulate cupy presence

        original_has_cupy = core_transformer.HAS_CUPY
        original_cp = getattr(core_transformer, 'cp', None)

        try:
            core_transformer.HAS_CUPY = True
            mock_cp = MagicMock()
            core_transformer.cp = mock_cp

            # Setup mock returns
            a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
            b_np = np.array([[1, 0], [0, 1]], dtype=np.float32)

            a_cp_arr = MagicMock()
            b_cp_arr = MagicMock()
            res_cp_arr = MagicMock()

            # Configure asarray to return our mock arrays
            # We need to distinguish calls. side_effect can be a list or function.
            # tiled_matmul calls asarray(a) then asarray(b)
            mock_cp.asarray.side_effect = [a_cp_arr, b_cp_arr]
            mock_cp.matmul.return_value = res_cp_arr
            mock_cp.asnumpy.return_value = np.matmul(a_np, b_np)

            # Test backend="cupy" with numpy inputs
            res = tiled_matmul(a_np, b_np, backend="cupy")

            # Verify calls
            self.assertEqual(mock_cp.asarray.call_count, 2)
            mock_cp.matmul.assert_called_with(a_cp_arr, b_cp_arr)
            mock_cp.asnumpy.assert_called_with(res_cp_arr)

            # Test correctness of result passed through
            np.testing.assert_allclose(res, np.matmul(a_np, b_np))

        finally:
            core_transformer.HAS_CUPY = original_has_cupy
            if original_cp:
                core_transformer.cp = original_cp
            else:
                del core_transformer.cp

    def test_tiled_matmul_gpu_real(self):
        # Only run if cupy is actually available
        if not HAS_CUPY:
            print("Skipping real GPU test as cupy is not installed")
            return

        import cupy as cp
        M, K, N = 128, 128, 128
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)

        # Test numpy inputs -> cupy backend -> numpy output
        res = tiled_matmul(a_np, b_np, backend="cupy")
        expected = np.matmul(a_np, b_np)
        np.testing.assert_allclose(res, expected, atol=1e-5)

        # Test cupy inputs -> cupy backend -> cupy output
        a_cp = cp.asarray(a_np)
        b_cp = cp.asarray(b_np)
        res_cp = tiled_matmul(a_cp, b_cp, backend="cupy")
        self.assertTrue(isinstance(res_cp, cp.ndarray))
        np.testing.assert_allclose(cp.asnumpy(res_cp), expected, atol=1e-5)

        # Test auto backend with cupy inputs
        res_auto = tiled_matmul(a_cp, b_cp, backend="auto")
        self.assertTrue(isinstance(res_auto, cp.ndarray))
        np.testing.assert_allclose(cp.asnumpy(res_auto), expected, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
