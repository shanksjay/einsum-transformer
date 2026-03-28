import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor
from core_transformer import tiled_matmul

def test_batched_matmul():
    np.random.seed(0)
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(2, 3, 5, 6).astype(np.float32)

    with ThreadPoolExecutor(max_workers=4) as executor:
        out1 = tiled_matmul(a, b, executor=executor)
        out2 = np.matmul(a, b)

        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)

def test_batched_matmul_a_2d():
    np.random.seed(0)
    a = np.random.randn(4, 5).astype(np.float32)
    b = np.random.randn(2, 3, 5, 6).astype(np.float32)

    with ThreadPoolExecutor(max_workers=4) as executor:
        out1 = tiled_matmul(a, b, executor=executor)
        out2 = np.matmul(a, b)

        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)

test_batched_matmul()
test_batched_matmul_a_2d()
print("Tests passed!")
