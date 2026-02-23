import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Adjust path to import core_transformer
sys.path.append(os.getcwd())
import core_transformer
from core_transformer import tiled_matmul

class TestTiledMatmul(unittest.TestCase):
    def setUp(self):
        self.a_2d = np.random.randn(128, 64).astype(np.float32)
        self.b_2d = np.random.randn(64, 128).astype(np.float32)

        self.a_3d = np.random.randn(4, 32, 64).astype(np.float32)
        self.b_3d = np.random.randn(4, 64, 32).astype(np.float32) # Batched B

        self.executor = ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        self.executor.shutdown()

    def test_2d_matmul_correctness(self):
        expected = np.matmul(self.a_2d, self.b_2d)
        result = tiled_matmul(self.a_2d, self.b_2d, block_size=32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batched_a_matmul_correctness(self):
        # a is 3D, b is 2D
        expected = np.matmul(self.a_3d, self.b_2d)
        result = tiled_matmul(self.a_3d, self.b_2d, block_size=32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batched_b_matmul_correctness(self):
        # a is 3D, b is 3D
        expected = np.matmul(self.a_3d, self.b_3d)
        result = tiled_matmul(self.a_3d, self.b_3d, block_size=32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batched_parallel_execution(self):
        # Trigger parallel path: total_ops > 1.5e8
        # M=100, K=2000, N=1000 -> 2e8 ops
        # Batch size 2
        a = np.random.randn(2, 100, 2000).astype(np.float32)
        b = np.random.randn(2, 2000, 1000).astype(np.float32)

        expected = np.matmul(a, b)

        # We assume result matches. We can't easily verify threads were used without logging/mocking executor,
        # but if logic is wrong it might crash or produce wrong result.
        # We patch compute_batch_slice or something? No, it's inside the function.
        # We can just verify correctness for now.

        # To verify it actually took the path, we can assert on the result shape/values.
        result = tiled_matmul(a, b, executor=self.executor)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_batched_parallel_execution_with_out(self):
        # M=100, K=2000, N=1000 -> 2e8 ops
        a = np.random.randn(2, 100, 2000).astype(np.float32)
        b = np.random.randn(2, 2000, 1000).astype(np.float32)
        expected = np.matmul(a, b)

        out = np.zeros_like(expected)
        result = tiled_matmul(a, b, executor=self.executor, out=out)

        # Verify result is returned and out is modified
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
        self.assertIs(result, out)

    def test_cupy_backend(self):
        # Mock core_transformer.cp and HAS_CUPY
        with patch('core_transformer.HAS_CUPY', True):
            with patch('core_transformer.cp', create=True) as mock_cp:
                # Setup mock behavior
                # define a mock array class
                class MockArray:
                    def __init__(self, data):
                        self.data = data
                    def get(self, out=None):
                        return self.data

                mock_cp.ndarray = MockArray

                def mock_array(x, copy=True):
                    return MockArray(x)

                mock_cp.array.side_effect = mock_array

                def mock_matmul(a, b, out=None):
                    res_data = np.matmul(a.data, b.data)
                    return MockArray(res_data)

                mock_cp.matmul.side_effect = mock_matmul

                # Run
                res = tiled_matmul(self.a_2d, self.b_2d, backend="cupy")

                # Check calls
                self.assertTrue(mock_cp.array.called)
                self.assertTrue(mock_cp.matmul.called)

                # Check result
                expected = np.matmul(self.a_2d, self.b_2d)
                np.testing.assert_allclose(res, expected, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
