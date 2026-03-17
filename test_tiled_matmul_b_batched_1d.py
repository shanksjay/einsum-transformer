import numpy as np
from core_transformer import tiled_matmul
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(4)

a = np.random.randn(4)
b = np.random.randn(2, 4, 5)

out = np.zeros((2, 1, 5))
res = tiled_matmul(a, b, executor=executor, out=out)
res2 = np.matmul(a[np.newaxis, np.newaxis, :], b)

assert np.allclose(res, res2)
print("1D batched test passed!")
