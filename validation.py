import numpy as np
from core_transformer import tiled_matmul

print("Validation basic batched:")
a = np.random.randn(2, 4, 8)
b = np.random.randn(2, 8, 4)
c1 = np.matmul(a, b)
c2 = tiled_matmul(a, b)
assert np.allclose(c1, c2)

print("Validation broadcast a:")
a = np.random.randn(4, 8)
b = np.random.randn(2, 8, 4)
c1 = np.matmul(a, b)
c2 = tiled_matmul(a, b)
assert np.allclose(c1, c2)

print("Validation broadcast b:")
a = np.random.randn(2, 4, 8)
b = np.random.randn(8, 4)
c1 = np.matmul(a, b)
c2 = tiled_matmul(a, b)
assert np.allclose(c1, c2)

print("Validation with out array:")
a = np.random.randn(2, 4, 8)
b = np.random.randn(2, 8, 4)
out = np.empty((2, 4, 4))
res = tiled_matmul(a, b, out=out)
assert res is out
assert np.allclose(np.matmul(a,b), out)

print("Batched Matmul Validation complete!")
