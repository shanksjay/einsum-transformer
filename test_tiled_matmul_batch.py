import numpy as np
import pytest
from core_transformer import tiled_matmul
from concurrent.futures import ThreadPoolExecutor

def test_batched_matmul_with_executor():
    a = np.random.rand(2, 3, 4, 5)
    b = np.random.rand(2, 3, 5, 6)

    with ThreadPoolExecutor(max_workers=4) as executor:
        res = tiled_matmul(a, b, executor=executor)

    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected)

def test_batched_matmul_without_executor():
    a = np.random.rand(2, 3, 4, 5)
    b = np.random.rand(2, 3, 5, 6)

    res = tiled_matmul(a, b)

    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected)

def test_batched_matmul_fallback_ndim_1():
    a = np.random.rand(5)
    b = np.random.rand(2, 3, 5, 6)

    res = tiled_matmul(a, b)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(res, expected)

if __name__ == '__main__':
    test_batched_matmul_with_executor()
    test_batched_matmul_without_executor()
    test_batched_matmul_fallback_ndim_1()
    print("All tests passed.")
