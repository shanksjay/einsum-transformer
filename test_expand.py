import numpy as np

a = np.random.randn(4)
b = np.random.randn(2, 4, 5)

res1 = np.matmul(a, b)
res2 = np.matmul(a[np.newaxis, np.newaxis, :], b)

print(res1.shape)
print(res2.shape)
