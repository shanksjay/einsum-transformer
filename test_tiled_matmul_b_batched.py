import numpy as np
from core_transformer import tiled_matmul
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(4)

a = np.random.randn(2, 3, 4000)
b = np.random.randn(2, 4000, 5000)

out = np.zeros((2, 3, 5000))
res = tiled_matmul(a, b, executor=executor, out=out)
assert np.allclose(res, a @ b)
print("Large batched test passed!")
