from core_transformer import tiled_matmul
import numpy as np
import time

A = np.random.randn(2048, 2048)
B = np.random.randn(2048, 2048)

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

t0 = time.time()
C = tiled_matmul(A, B, executor=executor)
print(time.time() - t0)

t0 = time.time()
C_np = np.matmul(A, B)
print(time.time() - t0)

print(np.allclose(C, C_np))
