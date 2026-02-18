import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import core_transformer
from core_transformer import tiled_matmul

class TestTiledMatmul(unittest.TestCase):
    def setUp(self):
        self.executor = core_transformer.ThreadPoolExecutor(max_workers=2)

    def test_matmul_2d_correctness(self):
        """Test standard 2D matrix multiplication correctness."""
        M, K, N = 128, 64, 32
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        # Force numpy backend to test our tiling implementation
        result = tiled_matmul(a, b, block_size=16, executor=self.executor, backend="numpy")

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_matmul_batched_correctness(self):
        """Test batched matrix multiplication correctness."""
        B, M, K, N = 5, 32, 16, 8
        a = np.random.rand(B, M, K).astype(np.float32)
        b = np.random.rand(B, K, N).astype(np.float32)

        # Determine expected behavior:
        # np.matmul broadcasts a over b if b is (K, N), or both if both have batch dims.
        # But tiled_matmul doc says: "a: [..., K], b: [K, N]" usually.
        # If I want to support full batched matmul (A @ B), I need to test that.
        # Currently tiled_matmul falls back to np.matmul if b.ndim > 2.

        expected = np.matmul(a, b)
        result = tiled_matmul(a, b, block_size=16, executor=self.executor, backend="numpy")

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_out_parameter(self):
        """Verify 'out' parameter usage."""
        M, K, N = 64, 32, 16
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        result = tiled_matmul(a, b, block_size=16, executor=self.executor, backend="numpy", out=out)

        self.assertIs(result, out)
        np.testing.assert_allclose(out, np.matmul(a, b), rtol=1e-5)

    def test_batched_parallel_trigger(self):
        """Test that parallel path triggers for large enough operations."""
        # Ops threshold is 1.5e8
        # Let's construct a case > 1.5e8
        # B=2, M=1000, K=1000, N=100 -> 2 * 10^8 ops
        B, M, K, N = 2, 1000, 1000, 100

        # Use small data but say it's float32 (memory 8MB for a, 0.8MB for b)
        a = np.random.rand(B, M, K).astype(np.float32)
        b = np.random.rand(B, K, N).astype(np.float32)

        # Mock executor to verify submit is called (proving parallel path)
        mock_executor = MagicMock()
        # Mock future result
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_executor.submit.return_value = mock_future
        # We also need _max_workers if code used it, but we changed to os.cpu_count()

        # We need to compute expected result for verification, but since we mock executor,
        # the computation won't happen inside tiled_matmul's futures unless we execute them.
        # But tiled_matmul waits for futures.

        # Wait, if I mock executor, the actual computation inside compute_batch_chunk won't run
        # unless I side_effect the submit to run the function.

        # Let's just use a real executor and trust coverage, or use a Spy.
        # Using a Spy on the real executor is better.

        with patch.object(self.executor, 'submit', wraps=self.executor.submit) as mock_submit:
            result = tiled_matmul(a, b, executor=self.executor, backend="numpy")

            # Verify correctness
            expected = np.matmul(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-4)

            # Verify parallel path was hit
            # We expect submit to be called.
            # Number of calls depends on chunking. B=2. Workers >= 1.
            # Chunk size = max(1, (2 + W - 1) // W).
            # If W=4 (default), chunk=1. Calls=2.
            # If W=large, chunk=1.
            self.assertTrue(mock_submit.call_count > 0, "Executor.submit should be called for large batched matmul")

    def test_cupy_backend_logic(self):
        """Test logic dispatch for CuPy backend using mocks."""
        # We need to patch core_transformer.cp and core_transformer.HAS_CUPY
        # If cupy is not installed, HAS_CUPY is False.

        with patch.object(core_transformer, 'HAS_CUPY', True):
            # Mock cupy module
            mock_cp = MagicMock()
            # Setup mock array to behave somewhat like an array but distinguishable
            class MockArray:
                def __init__(self, data): self.data = data
                @property
                def shape(self): return self.data.shape
                @property
                def ndim(self): return self.data.ndim
                @property
                def dtype(self): return self.data.dtype
                def get(self): return self.data # simulate getting data back to numpy

            # Set ndarray to be the class itself so isinstance works
            mock_cp.ndarray = MockArray
            mock_cp.array = MagicMock(side_effect=lambda x: MockArray(np.array(x)))
            mock_cp.matmul = MagicMock(return_value=MockArray(np.array([1])))

            # Patch 'cp' in core_transformer.
            # Use create=True because it might not exist if import failed in module
            with patch.object(core_transformer, 'cp', mock_cp, create=True):
                # Also verify 'backend="cupy"' triggers conversion
                a = np.random.rand(10, 10)
                b = np.random.rand(10, 10)

                # Case 1: backend="cupy" forcing conversion
                res = tiled_matmul(a, b, backend="cupy")

                # Check that cp.array was called to convert inputs
                self.assertEqual(mock_cp.array.call_count, 2)
                # Check that cp.matmul was called
                mock_cp.matmul.assert_called()

                # Reset mocks
                mock_cp.array.reset_mock()
                mock_cp.matmul.reset_mock()

                # Case 2: backend="auto" with numpy inputs -> should use numpy/tiled path, NOT cupy
                # Wait, existing logic for mlx/cupy usually is:
                # if backend="auto" and inputs are ON gpu, use gpu.
                # if backend="auto" and inputs are numpy, use numpy.
                # Let's verify this behavior is preserved/implemented.

                tiled_matmul(a, b, backend="auto")
                mock_cp.array.assert_not_called()
                mock_cp.matmul.assert_not_called()

if __name__ == '__main__':
    unittest.main()
