import numpy as np
from core_transformer import tiled_matmul
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
a = np.random.rand(4, 16, 32)
b = np.random.rand(4, 32, 64)
c = tiled_matmul(a, b, executor=executor)
assert np.allclose(c, np.matmul(a, b))
print("Batched matmul test passed!")
