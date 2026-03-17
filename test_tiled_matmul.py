import numpy as np
from core_transformer import tiled_matmul
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(4)

a = np.random.randn(2, 3, 4)
b = np.random.randn(2, 4, 5)

out = np.zeros((2, 3, 5))
res = tiled_matmul(a, b, executor=executor, out=out)
assert np.allclose(res, a @ b)
print("Basic batch test passed!")
