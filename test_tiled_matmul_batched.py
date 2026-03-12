import numpy as np
import os

class MockExecutor:
    def submit(self, fn, *args):
        class Future:
            def result(self):
                return fn(*args)
        return Future()

def tiled_matmul(a, b, executor=None, out=None):
    if b.ndim > 2:
        if executor is not None and a.ndim > 1:
            try:
                a_batch_shape = a.shape[:-2]
                b_batch_shape = b.shape[:-2]

                if a_batch_shape == b_batch_shape:
                    B_total = int(np.prod(a_batch_shape))
                    M = a.shape[-2]
                    K = a.shape[-1]
                    N = b.shape[-1]

                    a_flat = a.reshape(B_total, M, K)
                    b_flat = b.reshape(B_total, K, N)

                    out_shape = list(a.shape)
                    out_shape[-1] = N

                    if out is not None and out.shape == tuple(out_shape) and out.dtype == a.dtype:
                        res = out
                    else:
                        res = np.empty(out_shape, dtype=a.dtype)

                    res_flat = res.reshape(B_total, M, N)
                    shares = np.shares_memory(res, res_flat)

                    num_workers = os.cpu_count() or 4
                    chunk_size = max(1, B_total // num_workers)

                    def compute_batch_chunk(start_idx, end_idx):
                        with np.errstate(all='ignore'):
                            np.matmul(a_flat[start_idx:end_idx], b_flat[start_idx:end_idx], out=res_flat[start_idx:end_idx])

                    futures = []
                    for i in range(0, B_total, chunk_size):
                        end_idx = min(i + chunk_size, B_total)
                        futures.append(executor.submit(compute_batch_chunk, i, end_idx))

                    for f in futures:
                        f.result()

                    if not shares:
                        np.copyto(res, res_flat.reshape(out_shape))

                    return res
            except Exception:
                pass

        with np.errstate(all='ignore'):
            if a.ndim == 2 and b.ndim == 3 and a.shape[0] == b.shape[0]:
                res = np.matmul(a[:, np.newaxis, :], b).squeeze(1)
                if out is not None:
                    out[:] = res
                    return out
                return res
            return np.matmul(a, b, out=out)

a = np.random.rand(2, 4, 3, 5)
b = np.random.rand(2, 4, 5, 6)
exe = MockExecutor()
res1 = tiled_matmul(a, b, executor=exe)
res2 = np.matmul(a, b)
print("4D batch match:", np.allclose(res1, res2))

a = np.random.rand(2, 5)
b = np.random.rand(2, 5, 6)
res1 = tiled_matmul(a, b, executor=exe)
res2 = np.matmul(a[:, np.newaxis, :], b).squeeze(1)
print("1D vector and 3D tensor:", np.allclose(res1, res2))

a = np.random.rand(4, 5)
b = np.random.rand(2, 4, 5, 6)
res1 = tiled_matmul(a, b, executor=exe)
res2 = np.matmul(a, b)
print("No batch match fallback:", np.allclose(res1, res2))
