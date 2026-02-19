
import numpy as np
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
# Import tiled_matmul from core_transformer.py.
# Since core_transformer.py is in the root, I need to make sure I can import it.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from core_transformer import tiled_matmul

class TestTiledMatmul(unittest.TestCase):
    def test_2d_matmul(self):
        M, K, N = 1024, 1024, 1024
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        executor = ThreadPoolExecutor(max_workers=4)
        c_tiled = tiled_matmul(a, b, executor=executor, block_size=256)
        c_ref = np.matmul(a, b)

        np.testing.assert_allclose(c_tiled, c_ref, rtol=1e-4, atol=1e-4)
        print("2D Tiled Matmul passed")

    def test_batched_matmul_broadcast_a(self):
        # a: [B, M, K], b: [K, N] -> [B, M, N] (This is what tiled_matmul currently supports)
        B, M, K, N = 4, 128, 128, 128
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        executor = ThreadPoolExecutor(max_workers=4)
        c_tiled = tiled_matmul(a, b, executor=executor, block_size=64)
        c_ref = np.matmul(a, b)

        np.testing.assert_allclose(c_tiled, c_ref, rtol=1e-5, atol=1e-5)
        print("Batched A (broadcast B) passed")

    def test_batched_matmul_both(self):
        # a: [B, M, K], b: [B, K, N] -> [B, M, N]
        # Current tiled_matmul falls back to np.matmul for this case.
        B, M, K, N = 4, 128, 128, 128
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        executor = ThreadPoolExecutor(max_workers=4)
        # Verify it works (even if via fallback)
        c_tiled = tiled_matmul(a, b, executor=executor, block_size=64)
        c_ref = np.matmul(a, b)

        np.testing.assert_allclose(c_tiled, c_ref, rtol=1e-5, atol=1e-5)
        print("Batched A and B passed (fallback verify)")

    def test_batched_parallel_path(self):
        # a: [B, M, K], b: [B, K, N]
        # Make it large enough to trigger parallelism?
        # M, K, N = 256, 256, 256 -> ops = 256^3 = 1.6e7. B=20 -> 3.2e8 > 1.5e8.
        B, M, K, N = 20, 256, 256, 256
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        executor = ThreadPoolExecutor(max_workers=4)
        # This should trigger the new parallel path
        c_tiled = tiled_matmul(a, b, executor=executor)
        c_ref = np.matmul(a, b)

        np.testing.assert_allclose(c_tiled, c_ref, rtol=1e-4, atol=1e-4)
        print("Batched Parallel Path passed")

if __name__ == "__main__":
    unittest.main()
