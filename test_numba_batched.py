import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def _numba_batched_tiled_matmul(a, b, res, block_size):
    B, M, K = a.shape
    N = b.shape[2]

    for b_idx in nb.prange(B):
        for i in range(0, M, block_size):
            m_end = min(i + block_size, M)
            for j in range(0, N, block_size):
                n_end = min(j + block_size, N)

                # Numba supports @ for 2D slices
                res[b_idx, i:m_end, j:n_end] = a[b_idx, i:m_end, :] @ b[b_idx, :, j:n_end]

a = np.random.rand(2, 4, 3, 5)
b = np.random.rand(2, 4, 5, 6)

a_flat = a.reshape(8, 3, 5)
b_flat = b.reshape(8, 5, 6)
res = np.empty((8, 3, 6))

_numba_batched_tiled_matmul(a_flat, b_flat, res, 2)
res_orig = np.matmul(a, b).reshape(8, 3, 6)
print("Numba batched match:", np.allclose(res, res_orig))
