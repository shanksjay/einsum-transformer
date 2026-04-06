import numpy as np
from concurrent.futures import ThreadPoolExecutor

def tiled_matmul(a, b, block_size=None, executor=None, backend="auto", out=None):
    if b.ndim > 2:
        if a.ndim == 1:
            with np.errstate(all='ignore'):
                return np.matmul(a, b, out=out)

        if a.ndim > 2 and b.ndim > 2 and a.shape[:-2] != b.shape[:-2]:
            with np.errstate(all='ignore'):
                return np.matmul(a, b, out=out)

        if executor is None:
            with np.errstate(all='ignore'):
                return np.matmul(a, b, out=out)

        try:
            batch_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            B_total = int(np.prod(batch_shape))
            M_inner, K = a.shape[-2], a.shape[-1]
            K_b, N_inner = b.shape[-2], b.shape[-1]

            out_dtype = np.result_type(a, b)
            out_shape = list(batch_shape) + [M_inner, N_inner]

            if out is not None and out.shape == tuple(out_shape) and out.dtype == out_dtype:
                res = out
            else:
                res = np.empty(out_shape, dtype=out_dtype)

            a_b = np.broadcast_to(a, list(batch_shape) + [M_inner, K])
            b_b = np.broadcast_to(b, list(batch_shape) + [K_b, N_inner])

            a_flat = a_b.reshape(B_total, M_inner, K)
            b_flat = b_b.reshape(B_total, K_b, N_inner)
            res_flat = res.reshape(B_total, M_inner, N_inner)

            def compute_batch(idx):
                with np.errstate(all='ignore'):
                    np.matmul(a_flat[idx], b_flat[idx], out=res_flat[idx])

            futures = [executor.submit(compute_batch, i) for i in range(B_total)]
            for f in futures:
                f.result()

            if not np.shares_memory(res, res_flat):
                np.copyto(res, res_flat.reshape(res.shape))

            return res

        except ValueError:
            with np.errstate(all='ignore'):
                return np.matmul(a, b, out=out)

    a_shape = a.shape
    K = a_shape[-1]
    M = int(np.prod(a_shape[:-1]))
    N = b.shape[-1]

    total_ops = float(M) * K * N
    use_parallel = executor is not None and total_ops > 1.5e8
    if M < 32:
        use_parallel = False

    if not use_parallel and total_ops < 2e8:
        with np.errstate(all='ignore'):
            return np.matmul(a, b, out=out)

    try:
        a_flat = a.reshape(M, K)
    except:
        with np.errstate(all='ignore'):
            return np.matmul(a, b, out=out)

    out_shape = list(a_shape)
    out_shape[-1] = N

    out_dtype = np.result_type(a, b)

    if out is not None and out.shape == tuple(out_shape) and out.dtype == out_dtype:
        res = out
    else:
        res = np.empty(out_shape, dtype=out_dtype)

    res_flat = res.reshape(M, N)
    return res

a = np.random.rand(2, 4, 3)
b = np.random.rand(2, 3, 5)

with ThreadPoolExecutor(2) as exe:
    res1 = tiled_matmul(a, b, executor=exe)
    res2 = np.matmul(a, b)
    assert np.allclose(res1, res2)

a = np.random.rand(4, 3)
b = np.random.rand(2, 3, 5)

with ThreadPoolExecutor(2) as exe:
    res1 = tiled_matmul(a, b, executor=exe)
    res2 = np.matmul(a, b)
    assert np.allclose(res1, res2)

print("SUCCESS")
