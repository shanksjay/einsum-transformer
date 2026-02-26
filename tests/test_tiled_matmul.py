
import unittest
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_transformer import tiled_matmul

class TestTiledMatmul(unittest.TestCase):
    def test_standard_matmul_2d(self):
        """Test standard 2D matrix multiplication correctness."""
        M, K, N = 64, 64, 64
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        result = tiled_matmul(a, b, block_size=32, backend="numpy")

        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_tiled_matmul_serial(self):
        """Test tiled matmul in serial execution."""
        M, K, N = 128, 128, 128
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        # No executor -> serial
        result = tiled_matmul(a, b, block_size=32, backend="numpy")

        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_tiled_matmul_parallel(self):
        """Test tiled matmul in parallel execution."""
        # Use dimensions that exceed 1.5e8 ops to trigger parallel path
        # 600^3 = 2.16e8 > 1.5e8
        M, K, N = 600, 600, 600
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        executor = ThreadPoolExecutor(max_workers=2)

        result = tiled_matmul(a, b, block_size=200, executor=executor, backend="numpy")
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_matmul_broadcast(self):
        """Test batched matmul where A is batched and B is 2D (broadcasting)."""
        B, M, K, N = 4, 32, 32, 32
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        result = tiled_matmul(a, b, block_size=16, backend="numpy")

        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_batched_matmul_parallel(self):
        """Test fully batched matmul (A and B are batched)."""
        # Trigger parallel path: 4 * 256 * 512 * 512 = 268,435,456 > 1.5e8
        B, M, K, N = 4, 256, 512, 512
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        expected = np.matmul(a, b)
        executor = ThreadPoolExecutor(max_workers=2)
        result = tiled_matmul(a, b, block_size=128, executor=executor, backend="numpy")

        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_out_parameter(self):
        """Test that out parameter is respected."""
        M, K, N = 64, 64, 64
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        tiled_matmul(a, b, block_size=32, backend="numpy", out=out)
        expected = np.matmul(a, b)

        np.testing.assert_allclose(out, expected, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
