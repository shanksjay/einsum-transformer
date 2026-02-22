import unittest
import numpy as np
from unittest.mock import MagicMock, patch, ANY
import sys
from concurrent.futures import ThreadPoolExecutor

# Mock cupy before importing core_transformer if possible, but core_transformer imports at top level.
# We can use patch.dict to inject a mock cupy into sys.modules
mock_cp = MagicMock()
mock_cp.ndarray = MagicMock() # Should not be used for isinstance checks if we can help it, or we need to patch existing objects.
# Actually, core_transformer uses checks like HAS_CUPY.
# We will verify the logic by patching HAS_CUPY and sys.modules['cupy'] contextually or just checking the function logic.

from core_transformer import tiled_matmul, HAS_MLX, HAS_NUMBA

class TestTiledMatmul(unittest.TestCase):
    def test_basic_2d(self):
        M, K, N = 64, 64, 64
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        res = tiled_matmul(a, b, block_size=32)
        expected = a @ b
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_batched_3d(self):
        B, M, K, N = 4, 32, 64, 32
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        # Currently this falls back to np.matmul
        res = tiled_matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_batched_4d_broadcasting(self):
        # Case similar to Attention Scores: [B, H, L, D] @ [B, H, D, T]
        B, H, L, T, D = 2, 4, 16, 16, 32
        a = np.random.randn(B, H, L, D).astype(np.float32)
        b = np.random.randn(B, H, D, T).astype(np.float32)

        res = tiled_matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_batched_mismatched_batch_dims(self):
        # Case where batch dims don't match exactly (broadcasting)
        # a: [1, M, K], b: [B, K, N] -> Output [B, M, N]
        B, M, K, N = 4, 32, 64, 32
        a = np.random.randn(1, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        res = tiled_matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_cupy_backend_call(self):
        with patch('core_transformer.HAS_CUPY', True, create=True), \
             patch('core_transformer.cp', mock_cp, create=True):

            # Define a Mock class for ndarray so isinstance checks work
            class MockNdArray:
                def get(self, out=None):
                    if out is not None:
                        out[:] = res_np_back[:]
                        return out
                    return res_np_back

            # Assign this class to mock_cp.ndarray
            mock_cp.ndarray = MockNdArray

            # Setup mock behavior
            a_np = np.random.randn(10, 10).astype(np.float32)
            b_np = np.random.randn(10, 10).astype(np.float32)

            # Mock objects
            a_cp = MockNdArray()
            b_cp = MockNdArray()
            res_cp = MockNdArray()
            res_np_back = np.random.randn(10, 10).astype(np.float32) # Dummy result

            mock_cp.array.side_effect = lambda x: a_cp if x is a_np else (b_cp if x is b_np else MagicMock())
            mock_cp.matmul.return_value = res_cp

            # Run with backend="cupy"
            res = tiled_matmul(a_np, b_np, backend="cupy")

            # Verify cp.matmul was called
            mock_cp.matmul.assert_called()
            # Verify conversion happened (check call count at least)
            self.assertGreaterEqual(mock_cp.array.call_count, 2)
            # Verify result
            self.assertTrue(np.array_equal(res, res_np_back))

    def test_batched_parallel_execution(self):
        # Verify that parallel execution path is taken for large ops
        # We can force it by mocking executor or using a custom one.
        # But tiled_matmul creates tasks.

        B, M, K, N = 10, 128, 128, 128

        # Mock executor
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_executor.submit.return_value = mock_future

        # Total ops = 10 * 128 * 128 * 128 = 20,971,520 (approx 2e7)
        # Threshold is 1.5e8. We need larger inputs or force threshold lower.
        # We can't force threshold easily as it's hardcoded.
        # Increase B significantly.
        # B=100 -> 2e8 > 1.5e8.
        B_large = 100
        a_large = np.random.randn(B_large, M, K).astype(np.float32)
        b_large = np.random.randn(B_large, K, N).astype(np.float32)

        # Patch as_completed to avoid real waiting/errors on mocks
        with patch('core_transformer.as_completed', side_effect=lambda futures: futures):
            tiled_matmul(a_large, b_large, executor=mock_executor)

        # Verify submit called
        # B_large = 100
        self.assertEqual(mock_executor.submit.call_count, B_large)

if __name__ == '__main__':
    unittest.main()
