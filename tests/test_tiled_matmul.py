import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure core_transformer can be imported
sys.path.append(os.getcwd())
import core_transformer
from core_transformer import tiled_matmul

class TestTiledMatmul:
    def setup_method(self):
        # Reset any module-level modifications if needed
        pass

    def test_simple_matmul(self):
        """Test standard 2D matrix multiplication."""
        M, K, N = 64, 128, 64
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        result = tiled_matmul(a, b, block_size=32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_batched_matmul_3d(self):
        """Test 3D batched matrix multiplication (B, M, K) @ (B, K, N)."""
        B, M, K, N = 4, 32, 64, 32
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)

        expected = np.matmul(a, b)

        # Test with executor to trigger parallel path
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            result = tiled_matmul(a, b, block_size=32, executor=executor)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

        # Test without executor (fallback)
        result_serial = tiled_matmul(a, b, block_size=32)
        np.testing.assert_allclose(result_serial, expected, rtol=1e-5, atol=1e-5)

    def test_batched_broadcast(self):
        """Test broadcasting (B, M, K) @ (K, N)."""
        B, M, K, N = 4, 32, 64, 32
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        expected = np.matmul(a, b)
        result = tiled_matmul(a, b, block_size=32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_cupy_backend_mock(self):
        """Test backend='cupy' using mocks."""
        # Inject 'cp' and 'HAS_CUPY' into core_transformer if they don't exist
        if not hasattr(core_transformer, 'cp'):
             setattr(core_transformer, 'cp', None)
        if not hasattr(core_transformer, 'HAS_CUPY'):
             setattr(core_transformer, 'HAS_CUPY', False)

        mock_cp = MagicMock()
        # Setup mock behavior
        mock_cp.asarray.side_effect = lambda x: x
        mock_cp.matmul.side_effect = lambda x, y: np.matmul(x, y)
        mock_cp.asnumpy.side_effect = lambda x: x
        # Ensure mock array is not instance of np.ndarray
        class MockArray:
            def __init__(self, data): self.data = data
            @property
            def shape(self): return self.data.shape
            @property
            def dtype(self): return self.data.dtype
            def astype(self, t, copy=True): return self.data.astype(t)

        # We need mock_cp.ndarray to be a type
        mock_cp.ndarray = MockArray

        with patch('core_transformer.HAS_CUPY', True), \
             patch('core_transformer.cp', mock_cp):

            M, K, N = 32, 32, 32
            a = np.random.randn(M, K).astype(np.float32)
            b = np.random.randn(K, N).astype(np.float32)

            result = tiled_matmul(a, b, backend="cupy")

            # Verify cupy functions were called
            assert mock_cp.asarray.call_count >= 1
            assert mock_cp.matmul.called
            assert mock_cp.asnumpy.called
            np.testing.assert_allclose(result, np.matmul(a, b), rtol=1e-5)
