import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Adjust path to import core_transformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import core_transformer # Import module to patch globals if needed
from core_transformer import tiled_matmul, HAS_MLX, HAS_NUMBA

class TestTiledMatmul(unittest.TestCase):
    def test_2d_matmul(self):
        # standard 2D case
        M, K, N = 128, 64, 32
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        res = tiled_matmul(a, b, block_size=16)
        expected = a @ b
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    def test_batched_matmul_explicit_parallel(self):
        # Test the new batched path with matching dims
        # Use large enough dims to trigger parallel heuristic (total_ops > 1.5e8)
        # B=4, M=512, K=512, N=512 -> 4 * 134M = 536M Ops.

        B, M, K, N = 4, 512, 512, 512
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        real_executor = ThreadPoolExecutor(max_workers=2)
        res = tiled_matmul(a, b, executor=real_executor)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-4) # Use larger tolerance for float32 accumulation diffs
        real_executor.shutdown()

    def test_batched_matmul_fallback_broadcasting(self):
        # Case where batch dims don't match -> should fallback to np.matmul
        B, M, K, N = 4, 32, 64, 32
        # a: [B, M, K], b: [1, K, N]
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(1, K, N).astype(np.float32)

        res = tiled_matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    def test_cupy_backend(self):
        # Define MockGpuArray
        class MockGpuArray:
            def __init__(self, data):
                self.data = np.array(data)
            def get(self, out=None):
                if out is not None:
                    np.copyto(out, self.data)
                    return out
                return self.data
            @property
            def shape(self): return self.data.shape
            @property
            def ndim(self): return self.data.ndim
            @property
            def dtype(self): return self.data.dtype
            # Allow numpy logical ops if any
            def __array__(self): return self.data
            def __repr__(self): return f"MockGpu({self.data})"

        # Mock cupy module
        mock_cp = MagicMock()
        mock_cp.ndarray = MockGpuArray
        mock_cp.array.side_effect = lambda x: MockGpuArray(x)
        mock_cp.matmul.side_effect = lambda a, b: MockGpuArray(a.data @ b.data)
        mock_cp.asnumpy.side_effect = lambda x: x.data if isinstance(x, MockGpuArray) else np.array(x)
        mock_cp.copyto.side_effect = lambda dst, src: np.copyto(dst.data if isinstance(dst, MockGpuArray) else dst, src.data if isinstance(src, MockGpuArray) else src)

        # Inject cp if missing (since ImportError might have prevented its definition)
        added_cp = False
        if not hasattr(core_transformer, 'cp'):
            setattr(core_transformer, 'cp', mock_cp)
            added_cp = True

        try:
            with patch.object(core_transformer, 'cp', mock_cp), \
                 patch.object(core_transformer, 'HAS_CUPY', True):

                M, K, N = 32, 32, 32
                a = np.random.randn(M, K).astype(np.float32)
                b = np.random.randn(K, N).astype(np.float32)

                # 1. Test backend="cupy" with no out (implicit return)
                res = tiled_matmul(a, b, backend="cupy")
                # Should return a MockGpuArray
                self.assertIsInstance(res, MockGpuArray)
                np.testing.assert_allclose(res.data, a @ b, rtol=1e-5)

                # 2. Test backend="cupy" with numpy out
                out = np.zeros((M, N), dtype=np.float32)
                tiled_matmul(a, b, backend="cupy", out=out)
                np.testing.assert_allclose(out, a @ b, rtol=1e-5)

                # 3. Test backend="auto" with MockGpuArray inputs
                a_gpu = MockGpuArray(a)
                b_gpu = MockGpuArray(b)
                res_auto = tiled_matmul(a_gpu, b_gpu, backend="auto")
                self.assertIsInstance(res_auto, MockGpuArray)
                np.testing.assert_allclose(res_auto.data, a @ b, rtol=1e-5)
        finally:
            if added_cp:
                delattr(core_transformer, 'cp')

if __name__ == '__main__':
    unittest.main()
