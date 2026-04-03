import numpy as np
from core_transformer import tiled_matmul
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

# batched test
a = np.random.rand(2, 3, 4)
b = np.random.rand(2, 4, 5)
out = np.empty((2, 3, 5))

res = tiled_matmul(a, b, executor=executor, out=out)
print(res.shape)

# cupy test
import sys
from unittest.mock import MagicMock
sys.modules['cupy'] = MagicMock()
# Wait we can just write test cases later
