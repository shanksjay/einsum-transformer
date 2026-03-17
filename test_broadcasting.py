import numpy as np
from core_transformer import tiled_matmul

a = np.random.randn(4)
b = np.random.randn(2, 4, 5)

res1 = np.matmul(a, b)
res2 = tiled_matmul(a, b)

print(res1.shape)
print(res2.shape)

out = np.empty_like(res1)
res3 = tiled_matmul(a, b, out=out)
print(res3.shape)

assert np.allclose(res1, res2)
assert np.allclose(res1, res3)
print("Broadcasting test passed!")
