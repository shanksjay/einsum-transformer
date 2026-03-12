import numpy as np

def parallel_batched_matmul(a, b, executor=None, out=None):
    if b.ndim > 2:
        try:
            # Handle cases where a might have fewer dimensions
            a_batch_shape = a.shape[:-2] if a.ndim >= 2 else ()
            b_batch_shape = b.shape[:-2]

            # If a is 1D, it's treated as (1, K) for broadcasting in matmul
            a_is_1d = a.ndim == 1
            if a_is_1d:
                a_batch_shape = ()
                M_dim = 1
                K_dim = a.shape[0]
                a_eval = a.reshape(1, K_dim)
            else:
                M_dim = a.shape[-2]
                K_dim = a.shape[-1]
                a_eval = a

            batch_shape = np.broadcast_shapes(a_batch_shape, b_batch_shape)

            if executor is not None:
                a_bc = np.broadcast_to(a_eval, batch_shape + (M_dim, K_dim))
                b_bc = np.broadcast_to(b, batch_shape + (b.shape[-2], b.shape[-1]))

                B_total = int(np.prod(batch_shape))
                N_dim = b.shape[-1]

                a_flat = a_bc.reshape(B_total, M_dim, K_dim)
                b_flat = b_bc.reshape(B_total, b.shape[-2], N_dim)

                out_shape = batch_shape + ((N_dim,) if a_is_1d else (M_dim, N_dim))

                if out is not None and out.shape == out_shape and out.dtype == a.dtype:
                    res = out
                else:
                    res = np.empty(out_shape, dtype=a.dtype)

                res_flat = res.reshape(B_total, M_dim, N_dim)

                shares = np.shares_memory(res, res_flat)

                def compute_slice(idx):
                    with np.errstate(all='ignore'):
                        np.matmul(a_flat[idx], b_flat[idx], out=res_flat[idx])

                futures = [executor.submit(compute_slice, i) for i in range(B_total)]
                for f in futures:
                    f.result()

                if not shares:
                    np.copyto(res, res_flat.reshape(out_shape))

                return res
        except Exception as e:
            print("Error:", e)
            pass

    with np.errstate(all='ignore'):
        return np.matmul(a, b, out=out)

a = np.random.rand(2, 4, 3, 5)
b = np.random.rand(2, 1, 5, 6)
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
res1 = parallel_batched_matmul(a, b, executor)
res2 = np.matmul(a, b)
print("4D @ 4D:", np.allclose(res1, res2))

a = np.random.rand(5)
b = np.random.rand(2, 5, 6)
res1 = parallel_batched_matmul(a, b, executor)
res2 = np.matmul(a, b)
print("1D @ 3D:", np.allclose(res1, res2))
