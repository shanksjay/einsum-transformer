import unittest
import numpy as np
import os
import core_transformer
from concurrent.futures import ThreadPoolExecutor

class MockCuPyArray:
    def __init__(self, arr):
        self.arr = np.asarray(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def get(self, out=None):
        if out is not None:
            np.copyto(out, self.arr)
            return out
        return self.arr.copy()

class MockCuPy:
    ndarray = MockCuPyArray

    @staticmethod
    def asarray(arr):
        return MockCuPyArray(arr)

    @staticmethod
    def matmul(a, b, out=None):
        res = np.matmul(a.arr, b.arr)
        if out is not None:
            np.copyto(out.arr, res)
            return out
        return MockCuPyArray(res)

    @staticmethod
    def copyto(dst, src):
        np.copyto(dst.arr, src.arr)

setattr(core_transformer, 'cp', MockCuPy)
core_transformer.HAS_CUPY = True

class TestTiledMatmul(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(2)

    def tearDown(self):
        self.executor.shutdown()

    def test_cupy_backend_numpy_out(self):
        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)
        out = np.empty((10, 10))

        res = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)
        self.assertTrue(np.allclose(res, np.matmul(a, b)))
        self.assertTrue(res is out)

    def test_cupy_backend_cupy_out(self):
        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)
        out_np = np.empty((10, 10))
        out = MockCuPyArray(out_np)

        res = core_transformer.tiled_matmul(a, b, backend="cupy", out=out)
        self.assertTrue(np.allclose(res.arr, np.matmul(a, b)))
        self.assertTrue(res is out)

    def test_cupy_backend_no_out(self):
        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)

        res = core_transformer.tiled_matmul(a, b, backend="cupy")
        self.assertTrue(np.allclose(res, np.matmul(a, b)))

    def test_cupy_backend_cp_inputs(self):
        a_cp = MockCuPyArray(np.random.randn(10, 10))
        b_cp = MockCuPyArray(np.random.randn(10, 10))

        res = core_transformer.tiled_matmul(a_cp, b_cp, backend="cupy")
        self.assertTrue(np.allclose(res.arr, np.matmul(a_cp.arr, b_cp.arr)))
        self.assertTrue(isinstance(res, MockCuPyArray))

    def test_batched_tiled_matmul_3d(self):
        # We need large enough dims to trigger parallelization if we want to test parallel path
        # > 1.5e8 ops.
        # Let's mock the ops check or use large matrices.
        # Actually, let's just use small matrices and rely on basic batched logic
        # It falls back to np.matmul if ops < threshold, so let's mock the heuristic for testing.

        original_cpu_count = os.cpu_count
        try:
            os.cpu_count = lambda: 4
            a = np.random.randn(100, 200, 300)
            b = np.random.randn(100, 300, 250)

            # Need B_total * M_inner * K * N > 1.5e8
            # 100 * 200 * 300 * 250 = 1.5e9 > 1.5e8. Perfect.
            res = core_transformer.tiled_matmul(a, b, executor=self.executor)
            self.assertTrue(np.allclose(res, np.matmul(a, b)))
        finally:
            os.cpu_count = original_cpu_count

    def test_batched_tiled_matmul_4d(self):
        a = np.random.randn(10, 10, 200, 300)
        b = np.random.randn(10, 10, 300, 250)

        # 100 * 200 * 300 * 250 = 1.5e9 ops
        res = core_transformer.tiled_matmul(a, b, executor=self.executor)
        self.assertTrue(np.allclose(res, np.matmul(a, b)))

    def test_batched_tiled_matmul_with_out(self):
        a = np.random.randn(10, 10, 200, 300)
        b = np.random.randn(10, 10, 300, 250)
        out = np.zeros((10, 10, 200, 250))

        res = core_transformer.tiled_matmul(a, b, executor=self.executor, out=out)
        self.assertTrue(np.allclose(res, np.matmul(a, b)))
        self.assertTrue(res is out)

    def test_batched_tiled_matmul_shape_mismatch(self):
        a = np.random.randn(2, 4, 300)
        b = np.random.randn(1, 300, 250)

        res = core_transformer.tiled_matmul(a, b, executor=self.executor)
        self.assertTrue(np.allclose(res, np.matmul(a, b)))

if __name__ == "__main__":
    unittest.main()
