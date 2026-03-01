import re
import numpy as np
import os
import time
import copy
import threading
import functools
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from safetensors.numpy import load_file
from safetensors import safe_open
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


import platform

@functools.lru_cache(maxsize=1)
def _get_platform_block_size():
    """Determine optimal block size for tiled matmul based on platform."""
    sys_plat = platform.system()
    machine = platform.machine()

    if sys_plat == 'Darwin' and machine == 'arm64':
        # Apple Silicon (M1/M2/M3)
        # Optimal block size for AMX/NEON often around 2048-4096 depending on L2
        return 2048
    elif machine == 'x86_64':
        # Intel/AMD AVX512 usually prefers 512-1024 chunks to stay in L1/L2
        return 1024
    else:
        # Conservative default
        return 1024

def tiled_matmul(a, b, block_size=None, executor=None, backend="auto", out=None):
    """
    Perform matrix multiplication a @ b using tiling to reduce peak memory
    and improve stability. Supports parallel dispatch for chunks.

    a: [..., K]
    b: [K, N]
    Output: [..., N]
    backend: "auto", "mlx", "numba", "numpy"
    out: Optional output array (only used if shapes match)
    """
    # Force backend if specified
    if backend == "mlx" and HAS_MLX:
        if not isinstance(a, mx.array): a = mx.array(a)
        if not isinstance(b, mx.array): b = mx.array(b)
        return mx.matmul(a, b)

    if backend == "cupy" and HAS_CUPY:
        # Avoid isinstance check with np.ndarray since cp.ndarray inherits from it
        if not (type(a).__name__ == 'ndarray' and type(a).__module__ == 'cupy'): a = cp.array(a)
        if not (type(b).__name__ == 'ndarray' and type(b).__module__ == 'cupy'): b = cp.array(b)
        res = cp.matmul(a, b)
        if out is not None:
            if type(out).__name__ == 'ndarray' and type(out).__module__ == 'cupy':
                cp.copyto(out, res)
                return out
            else:
                return res.get(out=out)
        return res.get()

    # Auto GPU Path: If MLX is available and inputs are on GPU (or we want to use it), use it directly
    if (backend == "auto" and HAS_MLX) and (isinstance(a, mx.array) or isinstance(b, mx.array)):
        if not isinstance(a, mx.array): a = mx.array(a)
        if not isinstance(b, mx.array): b = mx.array(b)
        return mx.matmul(a, b)

    if (backend == "auto" and HAS_CUPY) and (type(a).__module__ == 'cupy' or type(b).__module__ == 'cupy'):
        if not (type(a).__name__ == 'ndarray' and type(a).__module__ == 'cupy'): a = cp.array(a)
        if not (type(b).__name__ == 'ndarray' and type(b).__module__ == 'cupy'): b = cp.array(b)
        res = cp.matmul(a, b)
        if out is not None:
            if type(out).__name__ == 'ndarray' and type(out).__module__ == 'cupy':
                cp.copyto(out, res)
                return out
            else:
                return res.get(out=out)
        return res.get()

    if block_size is None:
        block_size = _get_platform_block_size()

    a_shape = a.shape
    K = a_shape[-1]
    # Calculate M (product of leading dimensions of a)
    M = int(np.prod(a_shape[:-1]))
    N = b.shape[-1]

    # Batched matrix multiplication: if b.ndim > 2
    if b.ndim > 2:
        if executor is None or a.shape[:-2] != b.shape[:-2]:
            with np.errstate(all='ignore'):
                return np.matmul(a, b, out=out)

        batch_shape = a.shape[:-2]
        B_total = int(np.prod(batch_shape))
        M_inner, K_inner = a.shape[-2], a.shape[-1]
        N_inner = b.shape[-1]

        # Flatten batch dims
        a_flat_batch = a.reshape(B_total, M_inner, K_inner)
        b_flat_batch = b.reshape(B_total, K_inner, N_inner)

        out_shape = list(a.shape)
        out_shape[-1] = N_inner

        if out is not None and out.shape == tuple(out_shape) and out.dtype == a.dtype:
            res = out
        else:
            res = np.empty(out_shape, dtype=a.dtype)

        res_flat_batch = res.reshape(B_total, M_inner, N_inner)
        is_view = np.shares_memory(res, res_flat_batch)

        def compute_batch_element(b_idx):
            with np.errstate(all='ignore'):
                np.matmul(a_flat_batch[b_idx], b_flat_batch[b_idx], out=res_flat_batch[b_idx])

        futures = []
        for b_idx in range(B_total):
            futures.append(executor.submit(compute_batch_element, b_idx))

        for f in as_completed(futures):
            f.result()

        if not is_view:
            # If reshape created a copy, we must copy the results back into res
            res[...] = res_flat_batch.reshape(res.shape)

        return res

    # Heuristic: Parallelize if total work > threshold
    # Increased threshold to 1.5e8 (150M ops) to avoid thread overhead for medium sizes
    total_ops = float(M) * K * N
    use_parallel = executor is not None and total_ops > 1.5e8

    # Disable parallelization for small batch sizes (M < 32) as BLAS is faster
    if M < 32:
        use_parallel = False

    # If small work, just run standard matmul
    if not use_parallel and total_ops < 2e8:
        with np.errstate(all='ignore'):
            return np.matmul(a, b, out=out)

    # Flatten 'a' to [M, K] to allow uniform 2D tiling
    # Note: reshape returns a view if possible, avoiding copy
    try:
        a_flat = a.reshape(M, K)
    except:
        # Fallback if view not possible
        with np.errstate(all='ignore'):
            return np.matmul(a, b, out=out)

    out_shape = list(a_shape)
    out_shape[-1] = N

    if out is not None and out.shape == tuple(out_shape) and out.dtype == a.dtype:
        res = out
    else:
        res = np.empty(out_shape, dtype=a.dtype)

    res_flat = res.reshape(M, N)
    is_view = np.shares_memory(res, res_flat)

    use_numba = (backend == "numba" and HAS_NUMBA) or (backend == "auto" and HAS_NUMBA)

    if use_numba:
        _numba_tiled_matmul(a_flat, b, res_flat, block_size)
    else:
        def compute_block(m_start, m_end, n_start, n_end):
            # Perform matmul for the block
            # a_flat[m_start:m_end, :] @ b[:, n_start:n_end] -> res_flat[m_start:m_end, n_start:n_end]
            with np.errstate(all='ignore'):
                np.matmul(
                    a_flat[m_start:m_end, :],
                    b[:, n_start:n_end],
                    out=res_flat[m_start:m_end, n_start:n_end]
                )

        if use_parallel:
            futures = []
            # Tile over M (rows) and N (columns)
            for i in range(0, M, block_size):
                m_end = min(i + block_size, M)
                for j in range(0, N, block_size):
                    n_end = min(j + block_size, N)
                    futures.append(executor.submit(compute_block, i, m_end, j, n_end))

            # Wait for all tasks to complete
            for f in as_completed(futures):
                f.result()
        else:
            # Serial execution
            for i in range(0, M, block_size):
                m_end = min(i + block_size, M)
                for j in range(0, N, block_size):
                    n_end = min(j + block_size, N)
                    compute_block(i, m_end, j, n_end)

    if not is_view:
        # If reshape created a copy, we must copy the results back into res
        res[...] = res_flat.reshape(res.shape)

    return res

if HAS_NUMBA:
    @nb.njit(parallel=True, fastmath=True)
    def _numba_tiled_matmul(a, b, res, block_size):
        # a: [M, K], b: [K, N], res: [M, N]
        M, K = a.shape
        N = b.shape[1]
        # Parallelize outer loops over blocks
        # Numba's parallel loop handling is often better than manual chunking
        # We iterate over blocks to keep memory locality
        for i in nb.prange(0, M, block_size):
            m_end = min(i + block_size, M)
            for j in range(0, N, block_size):
                n_end = min(j + block_size, N)
                # Compute block: res[i:m_end, j:n_end] += a[i:m_end, :] @ b[:, j:n_end]
                # Since res is empty, we assign directly.
                # However, np.dot/matmul inside njit might not support out= argument or slicing assignment perfectly with BLAS
                # But typical usage:
                res[i:m_end, j:n_end] = a[i:m_end, :] @ b[:, j:n_end]
else:
    _numba_tiled_matmul = None


class TransformerConfig:
    def __init__(self, cfg):
        # Basic input validation
        required_keys = ["batch_size", "seq_len", "vocab_size", "d_model", "n_heads", "d_ff", "n_layers"]
        for k in required_keys:
            if k not in cfg:
                raise ValueError(f"Missing required config key: {k}")

        self.arch = cfg.get("arch", "custom")
        self.batch_size = cfg["batch_size"]
        self.seq_len = cfg["seq_len"]
        self.vocab_size = cfg["vocab_size"]
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg.get("n_kv_heads", self.n_heads)
        self.d_ff = cfg["d_ff"]
        self.n_layers = cfg["n_layers"]
        self.rope_base = cfg.get("rope_base", 10000.0)
        self.block_size = cfg.get("block_size", 16)

        self.use_gqa = cfg.get("use_gqa", False)
        self.use_mqa = cfg.get("use_mqa", False)
        self.use_moe = cfg.get("use_moe", False)

        self.num_experts = cfg.get("num_experts", 4)
        self.top_k = cfg.get("top_k", 2)

        # Generation config
        self.max_new_tokens = cfg.get("max_new_tokens", 10)
        self.temperature = cfg.get("temperature", 1.0)
        self.top_k_sample = cfg.get("top_k_sample", 50)

        self.quantized = cfg.get("quantized", False)  # legacy; prefer quantization
        self.prefill_only = cfg.get("prefill_only", False)

        # Speculative decoding
        self.speculative = cfg.get("speculative", False)
        self.draft_layers = cfg.get("draft_layers", 2)

        # Mixed-precision quantization: "none" | "W8A16" | "W8A8" | "W4A16" | "W4A4" | "W8A4" etc. (W=weight bits, A=activation bits)
        q = cfg.get("quantization", "none")
        if isinstance(q, str) and q.lower() in ("none", "off", ""):
            self.quantization = "none"
            self.quantization_weight_bits = 0
            self.quantization_activation_bits = 0
        else:
            m = re.match(r"W(\d+)A(\d+)", str(q), re.I) if isinstance(q, str) else None
            if m:
                self.quantization = str(q).upper()
                self.quantization_weight_bits = int(m.group(1))
                self.quantization_activation_bits = int(m.group(2))
            elif cfg.get("quantized", False):
                self.quantization = "W8A16"
                self.quantization_weight_bits = 8
                self.quantization_activation_bits = 16
            else:
                self.quantization = "none"
                self.quantization_weight_bits = 0
                self.quantization_activation_bits = 0

        self.weight_file = cfg.get("weight_file", None)
        self.hf_repo_id = cfg.get("hf_repo_id", None)
        self.cache_dir = cfg.get("cache_dir", str(Path.home() / ".cache" / "huggingface" / "hub"))
        self.weight_dtype = cfg.get("weight_dtype", "float32")
        self.strict_load = cfg.get("strict_load", True)

        self.use_lora = cfg.get("use_lora", False)  # Master switch to enable/disable LoRA
        self.lora_rank = cfg.get("lora_rank", 4)
        self.lora_alpha = cfg.get("lora_alpha", 1.0)
        self.lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
        self.lora_merge_after_training = cfg.get("lora_merge_after_training", False)

        # Backend configuration
        self.backend = cfg.get("backend", "auto")

        self.d_head = self.d_model // self.n_heads


class LazyWeightManager:
    def __init__(self, weight_paths, target_dtype="float32", max_cache_size=32):
        self.handles = {}
        self.lock = threading.Lock()
        self.tensor_to_shard = {}
        self.target_dtype = target_dtype
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()
        # Persistent cache for tensors not backed by files (e.g. generated LoRA adapters)
        self.persistent_cache = {}
        
        print(f"[INFO] Initializing LazyWeightManager with {len(weight_paths)} shards (max_cache_size={max_cache_size})...")
        for path in weight_paths:
            # safe_open uses mmap, keeping handles open is efficient
            handle = safe_open(path, framework="np")
            self.handles[path] = handle
            for key in handle.keys():
                if key in self.tensor_to_shard:
                    continue
                self.tensor_to_shard[key] = path

    def __getitem__(self, key):
        with self.lock:
            if key in self.persistent_cache:
                return self.persistent_cache[key]

            if key in self.cache:
                # Move to end (LRU)
                val = self.cache.pop(key)
                self.cache[key] = val
                return val
        
        if key not in self.tensor_to_shard:
            raise KeyError(f"Tensor {key} not found in any shard.")
        
        path = self.tensor_to_shard[key]
        handle = self.handles[path]
        
        # Initial fetch - handle bfloat16 defensively
        try:
            data = handle.get_tensor(key)
        except TypeError as e:
            if "bfloat16" in str(e):
                # If numpy doesn't understand bfloat16, we might need ml_dtypes
                print(f"[ERROR] bfloat16 detected but not supported by numpy. Error: {e}")
                raise
            raise

        # Casting logic
        if self.target_dtype == "original":
            res = data
        elif self.target_dtype == "bfloat16":
            if ml_dtypes:
                res = data.astype(ml_dtypes.bfloat16)
            else:
                res = data.astype(np.float32)
        elif self.target_dtype == "float16":
            res = data.astype(np.float16)
        else:
            res = data.astype(np.float32)
            
        with self.lock:
            self.cache[key] = res
            # Evict if necessary
            if self.max_cache_size and len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)
        return res

    def __setitem__(self, key, value):
        with self.lock:
            # Store modified or new weights in persistent cache to avoid eviction
            # and ensure training updates are preserved.
            self.persistent_cache[key] = value

            # If it was in the LRU cache, remove it as it's now in persistent
            if key in self.cache:
                del self.cache[key]

    def purge_cache(self):
        print(f"[INFO] Purging weight cache ({len(self.cache)} tensors removed)")
        self.cache.clear()

    def get_cache_memory_mb(self):
        b = sum(v.nbytes for v in self.cache.values() if isinstance(v, np.ndarray))
        b += sum(v.nbytes for v in self.persistent_cache.values() if isinstance(v, np.ndarray))
        return b / (1024 * 1024)

    def __contains__(self, key):
        return key in self.tensor_to_shard or key in self.cache or key in self.persistent_cache

    def keys(self):
        k = set(self.tensor_to_shard.keys())
        k.update(self.cache.keys())
        k.update(self.persistent_cache.keys())
        return list(k)


class DictWeightManager:
    """In-memory weight manager backed by a dict. Used for random init and for _from_weights (e.g. Ray workers)."""
    def __init__(self, weights_dict):
        self.cache = weights_dict

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def keys(self):
        return self.cache.keys()

    def purge_cache(self):
        pass

    def get_cache_memory_mb(self):
        b = sum(v.nbytes for v in self.cache.values() if isinstance(v, np.ndarray))
        return b / (1024 * 1024)


class EinsumTransformer:
    def __init__(self, cfg: TransformerConfig, *, _from_weights=None, _E_name=None, _layer_weights=None):
        self.cfg = cfg
        self.weights = None  # LazyWeightManager or DictWeightManager
        self.cache = [{"K": None, "V": None} for _ in range(cfg.n_layers)]
        self.cur_pos = 0
        self._activations = {}
        self._grads = {}
        self.training = False
        self._draft_model = None
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        # LoRA configuration defaults
        self.use_lora = getattr(self.cfg, "use_lora", False)
        self.lora_rank = getattr(self.cfg, "lora_rank", 4)
        self.lora_alpha = getattr(self.cfg, "lora_alpha", 1.0)
        self.lora_target_modules = getattr(self.cfg, "lora_target_modules", ["q_proj", "v_proj"])
        self.lora_merge_after_training = getattr(self.cfg, "lora_merge_after_training", False)

        # Optimization state
        self.optimized_layers = None
        self.optimized_params = None
        self.kv_cache_opt = None
        self.cos_cached = None
        self.sin_cached = None

        if _from_weights is not None:
            if _E_name is None or _layer_weights is None:
                self.weights = DictWeightManager(_from_weights) if isinstance(_from_weights, dict) else _from_weights
                self._bind_weights()
            else:
                self.weights = DictWeightManager(_from_weights) if isinstance(_from_weights, dict) else _from_weights
                self.E_name = _E_name
                self.layer_weights = _layer_weights
            return

        if cfg.weight_file:
            print(f"[INFO] weight_file specified: {cfg.weight_file}, attempting to load weights...")
            try:
                self._load_weights(cfg.weight_file)
            except Exception as e:
                if "Checkpoint architecture does not match config" in str(e):
                    raise
                print(f"[WARNING] Failed to load weights from {cfg.weight_file}: {e}")
                print(f"[INFO] Falling back to random weight initialization...")
                self._init_weights_random()
        else:
            print(f"[INFO] No weight_file specified, initializing random weights...")
            self._init_weights_random()
        if getattr(self.cfg, "quantization", "none") != "none":
            quant_bits_w = getattr(self.cfg, 'quantization_weight_bits', 0)
            quant_bits_a = getattr(self.cfg, 'quantization_activation_bits', 0)
            print(f"[INFO] Mixed-precision quantization: {self.cfg.quantization} (W{quant_bits_w} A{quant_bits_a})")

        self._init_rope()

    def _should_parallelize(self, total_ops):
        """Check if parallel execution is beneficial based on total FLOPs."""
        # Threshold: 1.5e8 ops (consistent with tiled_matmul heuristic)
        return self.executor is not None and total_ops > 1.5e8

    def _init_rope(self):
        """Precompute RoPE tables for the maximum sequence length."""
        c = self.cfg
        head_dim = c.d_head
        # Inverse frequencies
        inv_freq = 1.0 / (c.rope_base ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
        # Position indices
        t = np.arange(c.seq_len, dtype=np.float32)
        # Outer product: (seq_len, head_dim/2)
        freqs = np.outer(t, inv_freq)
        # Cache cos/sin values as float32 to save memory and match activation dtype usually
        self.rope_cache = {
            "cos": np.cos(freqs).astype(np.float32),
            "sin": np.sin(freqs).astype(np.float32)
        }

    def clear_cache(self):
        self.cache = [{"K": None, "V": None} for _ in range(self.cfg.n_layers)]
        self.cur_pos = 0
        if self.kv_cache_opt is not None:
            self.kv_cache_opt.fill(0)

    def _load_weights(self, path):
        print(f"[INFO] Attempting to load weights from: {path}")
        print(f"[INFO] hf_repo_id: {self.cfg.hf_repo_id}")
        
        if os.path.exists(path):
            print(f"[INFO] Loading safetensors from local path: {path}")
        else:
            if self.cfg.hf_repo_id:
                from huggingface_hub import hf_hub_download
                filename = os.path.basename(path)
                print(f"[INFO] File not found locally. Checking HuggingFace Hub cache for: {filename}")
                try:
                    path = hf_hub_download(
                        repo_id=self.cfg.hf_repo_id,
                        filename=filename,
                        cache_dir=self.cfg.cache_dir,
                        local_files_only=True
                    )
                    print(f"[INFO] Found in local cache: {path}")
                except Exception as e:
                    print(f"[INFO] File not found in local cache ({e}). Attempting to download from HF Hub: {self.cfg.hf_repo_id}")
                    try:
                        path = hf_hub_download(
                            repo_id=self.cfg.hf_repo_id,
                            filename=filename,
                            cache_dir=self.cfg.cache_dir,
                            token=os.environ.get("HUGGING_FACE_HUB_TOKEN")
                        )
                        print(f"[INFO] Downloaded to: {path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to download from HF Hub: {e}")
                        raise
            else:
                raise FileNotFoundError(f"Weight file not found: {path}")

        print(f"[INFO] Initial loading path: {path}")
        abs_path = os.path.abspath(path)
        weight_paths = [abs_path]
        weight_dir = os.path.dirname(abs_path)
        
        print(f"[INFO] Looking for additional shards in {weight_dir}...")
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                f_path = os.path.abspath(os.path.join(weight_dir, f))
                if f_path not in weight_paths:
                    print(f"[INFO] Found additional shard: {f}")
                    weight_paths.append(f_path)

        self.weights = LazyWeightManager(weight_paths, target_dtype=self.cfg.weight_dtype)
        self._init_lora_weights()
        self._bind_weights()
        self._adjust_cache_size()
        self._validate_loaded_shapes()
        self._print_per_layer_dtype_breakdown()

    def _init_lora_weights(self):
        """Initialize random LoRA adapter weights and inject them into the weight manager."""
        if not self.use_lora or not self.cfg.lora_target_modules:
            return

        c = self.cfg
        r = self.lora_rank
        print(f"[INFO] Initializing LoRA adapters (rank={r}) for modules: {c.lora_target_modules}")
        
        for i in range(c.n_layers):
            if "q_proj" in c.lora_target_modules:
                A = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                B = np.random.randn(r, c.d_model).astype(np.float32) * 0.02
                self.weights[f"lora_A_q_{i}"] = A
                self.weights[f"lora_B_q_{i}"] = B
                
            if "v_proj" in c.lora_target_modules:
                d_kv = c.n_kv_heads * c.d_head
                A = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                B = np.random.randn(r, d_kv).astype(np.float32) * 0.02
                self.weights[f"lora_A_v_{i}"] = A
                self.weights[f"lora_B_v_{i}"] = B

    def _init_weights_random(self):
        """Initialize random weights and wrap them in a dict-based LazyWeightManager interface."""
        c = self.cfg
        print("[INFO] Initializing random weights (no pre-trained model)...")
        
        weights_dict = {}
        weights_dict["E"] = np.random.randn(c.vocab_size, c.d_model).astype(np.float32) * 0.02
        self.E_name = "E"
        
        self.layer_weights = []
        for i in range(c.n_layers):
            d_kv = c.n_kv_heads * c.d_head
            weights_dict[f"W_q_{i}"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W_k_{i}"] = np.random.randn(d_kv, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W_v_{i}"] = np.random.randn(d_kv, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W_o_{i}"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * 0.02

            if self.use_lora:
                r = self.lora_rank
                if "q_proj" in self.lora_target_modules:
                    weights_dict[f"lora_A_q_{i}"] = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                    weights_dict[f"lora_B_q_{i}"] = np.random.randn(r, c.d_model).astype(np.float32) * 0.02
                if "v_proj" in self.lora_target_modules:
                    weights_dict[f"lora_A_v_{i}"] = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                    weights_dict[f"lora_B_v_{i}"] = np.random.randn(r, d_kv).astype(np.float32) * 0.02
            
            if c.use_moe:
                weights_dict[f"gate_{i}"] = np.random.randn(c.num_experts, c.d_model).astype(np.float32) * 0.02
                experts = []
                for e in range(c.num_experts):
                    weights_dict[f"W1_{i}_{e}"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * 0.02
                    weights_dict[f"W2_{i}_{e}"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * 0.02
                    weights_dict[f"W3_{i}_{e}"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * 0.02
                    experts.append({"W1": f"W1_{i}_{e}", "W2": f"W2_{i}_{e}", "W3": f"W3_{i}_{e}"})
            else:
                weights_dict[f"W1_{i}"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * 0.02
                weights_dict[f"W2_{i}"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * 0.02
                weights_dict[f"W3_{i}"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * 0.02
            
            weights_dict[f"norm1_{i}"] = np.ones(c.d_model).astype(np.float32)
            weights_dict[f"norm2_{i}"] = np.ones(c.d_model).astype(np.float32)
            
            layer = {
                "W_q": f"W_q_{i}", "W_k": f"W_k_{i}", "W_v": f"W_v_{i}", "W_o": f"W_o_{i}",
                "W1": f"W1_{i}" if not c.use_moe else None,
                "W2": f"W2_{i}" if not c.use_moe else None,
                "W3": f"W3_{i}" if not c.use_moe else None,
                "norm1": f"norm1_{i}", "norm2": f"norm2_{i}",
                "experts": experts if c.use_moe else None,
                "gate": f"gate_{i}" if c.use_moe else None
            }
            if self.use_lora:
                if "q_proj" in self.lora_target_modules:
                    layer["lora_A_q"], layer["lora_B_q"] = f"lora_A_q_{i}", f"lora_B_q_{i}"
                if "v_proj" in self.lora_target_modules:
                    layer["lora_A_v"], layer["lora_B_v"] = f"lora_A_v_{i}", f"lora_B_v_{i}"
            self.layer_weights.append(layer)
        
        self.weights = DictWeightManager(weights_dict)
        self._print_per_layer_dtype_breakdown()

    def _print_per_layer_dtype_breakdown(self):
        """Print per-layer operation breakdown by tensor datatype: source (original) → target (quantized). Includes memory saving."""
        c = self.cfg
        wb = getattr(c, "quantization_weight_bits", 0)
        ab = getattr(c, "quantization_activation_bits", 0)
        wd = getattr(c, "weight_dtype", "float32")
        # Dimensions used for memory calculations:
        V = c.vocab_size             # Vocabulary size
        d = c.d_model                # Model dimension (hidden size)
        d_ff = c.d_ff                # Feed-forward network intermediate dimension
        B = c.batch_size             # Batch size
        T = c.seq_len                # Sequence length (context window)
        n_layers = c.n_layers        # Total number of transformer layers
        d_kv = c.n_kv_heads * (c.d_model // c.n_heads) # Total dimension for Key/Value heads

        def weight_str():
            if wb == 8: return f"{wd}→int8"
            if wb == 4: return f"{wd}→int4"
            return wd

        def act_str():
            if ab == 16: return "float32→fp16"
            if ab == 8: return "float32→int8"
            if ab == 4: return "float32→int4"
            return "float32"

        def mb(b): return b / (1024 * 1024)

        # Source (Full precision) byte size
        src_bytes = 4 # float32
        if wd == "float16": src_bytes = 2
        elif wd == "bfloat16": src_bytes = 2

        # Quantized weight byte size
        q_bytes = src_bytes
        if wb == 8: q_bytes = 1
        elif wb == 4: q_bytes = 0.5

        # Activation byte size (simulated)
        a_src_bytes = 4 # float32
        a_q_bytes = a_src_bytes
        if ab == 16: a_q_bytes = 2
        elif ab == 8: a_q_bytes = 1
        elif ab == 4: a_q_bytes = 0.5

        def calc_mem(w_count, a_count, w_b, a_b):
            return mb(w_count * w_b + a_count * a_b)

        # Embedding: Weights = V * d; Activations = B * T * d
        emb_w = V * d
        emb_a = B * T * d
        emb_before = calc_mem(emb_w, emb_a, src_bytes, a_src_bytes)
        emb_after = calc_mem(emb_w, emb_a, q_bytes, a_q_bytes)

        # RMSNorm: 2 norms per layer, each with 'd' parameters
        rms_w = 2 * d * n_layers
        rms_a = 0 # RMSNorm activations usually small or same precision
        rms_before = calc_mem(rms_w, rms_a, src_bytes, a_src_bytes)
        rms_after = calc_mem(rms_w, rms_a, q_bytes, a_src_bytes)

        # Attention: Q,K,V,O projections + Attention Matrix (B, H, T, T)
        # We account for input (B,T,d), QKV projections (3*B,T,d), and the attention scores matrix (B,H,T,T).
        attn_w = (2 * d * d + 2 * d_kv * d) * n_layers
        attn_a = (4 * B * T * d + B * c.n_heads * T * T) * n_layers
        attn_before = calc_mem(attn_w, attn_a, src_bytes, a_src_bytes)
        attn_after = calc_mem(attn_w, attn_a, q_bytes, a_q_bytes)

        # FFN: SwiGLU uses 3 projections.
        # Activations: Input (B,T,d) + W1/W2/Gated intermediates (3 * B,T,d_ff)
        ffn_w = 3 * d * d_ff * n_layers
        ffn_a = (B * T * d + 3 * B * T * d_ff) * n_layers
        ffn_before = calc_mem(ffn_w, ffn_a, src_bytes, a_src_bytes)
        ffn_after = calc_mem(ffn_w, ffn_a, q_bytes, a_q_bytes)

        def fmt_mb(x): return f"{x:,.3f} MB" if x > 0 else "—"

        print(f"[INFO] Per-layer operation breakdown (by tensor datatype, source→target) and memory usage:")
        print(f"  {'Operation':<18} {'Weights':<18} {'Activations':<18} {'Before':<14} {'After':<14} {'Mem save':<14}")
        print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*14} {'-'*14} {'-'*14}")
        print(f"  {'Embedding':<18} {weight_str():<18} {act_str():<18} {fmt_mb(emb_before):<14} {fmt_mb(emb_after):<14} {fmt_mb(emb_before - emb_after):<14}")
        print(f"  {'RMSNorm (×2)':<18} {weight_str():<18} {'float32':<18} {fmt_mb(rms_before):<14} {fmt_mb(rms_after):<14} {fmt_mb(rms_before - rms_after):<14}")
        print(f"  {'Attn (Q,K,V,O)':<18} {weight_str():<18} {act_str():<18} {fmt_mb(attn_before):<14} {fmt_mb(attn_after):<14} {fmt_mb(attn_before - attn_after):<14}")
        print(f"  {'FFN (W1,W2,W3)':<18} {weight_str():<18} {act_str():<18} {fmt_mb(ffn_before):<14} {fmt_mb(ffn_after):<14} {fmt_mb(ffn_before - ffn_after):<14}")
        if self.use_lora:
            print(f"  {'LoRA (A,B)':<18} {'float32':<18} {'-':<18} {'—':<14} {'—':<14} {'—':<14}")
        print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*14} {'-'*14} {'-'*14}")
        total_before = emb_before + rms_before + attn_before + ffn_before
        total_after = emb_after + rms_after + attn_after + ffn_after
        print(f"  {'Total (all ' + str(n_layers) + ' layers)':<18} {'':<18} {'':<18} {fmt_mb(total_before):<14} {fmt_mb(total_after):<14} {fmt_mb(total_before - total_after):<14}")

    def _bind_weights(self):
        c = self.cfg
        def must(name):
            if name not in self.weights:
                if c.strict_load: raise KeyError(f"Missing tensor: {name}")
                print(f"[WARN] Missing tensor: {name}")
                return None
            return name

        def maybe(name):
            return name if name in self.weights else None

        self.E_name = must("model.embed_tokens.weight") or must("transformer.wte.weight")
        self.layer_weights = []
        for i in range(c.n_layers):
            prefix = f"model.layers.{i}."
            w_moe = None
            if c.use_moe:
                w_moe = []
                for e in range(c.num_experts):
                    e_prefix = prefix + f"block_sparse_moe.experts.{e}."
                    w_moe.append({
                        "W1": must(e_prefix + "w1.weight"),
                        "W2": must(e_prefix + "w2.weight"),
                        "W3": must(e_prefix + "w3.weight"),
                    })
                gate = must(prefix + "block_sparse_moe.gate.weight")
            else:
                W1 = must(prefix + "mlp.gate_proj.weight")
                W2 = must(prefix + "mlp.up_proj.weight")
                W3 = must(prefix + "mlp.down_proj.weight")
                gate = None

            layer = {
                "W_q": must(prefix + "self_attn.q_proj.weight"),
                "W_k": must(prefix + "self_attn.k_proj.weight"),
                "W_v": must(prefix + "self_attn.v_proj.weight"),
                "W_o": must(prefix + "self_attn.o_proj.weight"),
                "lora_A_q": maybe(prefix + "self_attn.q_proj.lora_A.weight") if self.use_lora and "q_proj" in self.lora_target_modules else None,
                "lora_B_q": maybe(prefix + "self_attn.q_proj.lora_B.weight") if self.use_lora and "q_proj" in self.lora_target_modules else None,
                "lora_A_v": maybe(prefix + "self_attn.v_proj.lora_A.weight") if self.use_lora and "v_proj" in self.lora_target_modules else None,
                "lora_B_v": maybe(prefix + "self_attn.v_proj.lora_B.weight") if self.use_lora and "v_proj" in self.lora_target_modules else None,
                "W1": W1 if not c.use_moe else None,
                "W2": W2 if not c.use_moe else None,
                "W3": W3 if not c.use_moe else None,
                "experts": w_moe, "gate": gate,
                "norm1": must(prefix + "input_layernorm.weight"),
                "norm2": must(prefix + "post_attention_layernorm.weight")
            }
            if self.use_lora:
                if layer["lora_A_q"] is None:
                    if f"lora_A_q_{i}" not in self.weights:
                        self.weights[f"lora_A_q_{i}"] = np.random.randn(c.d_model, self.lora_rank).astype(np.float32) * 0.02
                        self.weights[f"lora_B_q_{i}"] = np.random.randn(self.lora_rank, c.d_model).astype(np.float32) * 0.02
                    layer["lora_A_q"], layer["lora_B_q"] = f"lora_A_q_{i}", f"lora_B_q_{i}"
                if layer["lora_A_v"] is None:
                    if f"lora_A_v_{i}" not in self.weights:
                        d_kv = c.n_kv_heads * c.d_head
                        self.weights[f"lora_A_v_{i}"] = np.random.randn(c.d_model, self.lora_rank).astype(np.float32) * 0.02
                        self.weights[f"lora_B_v_{i}"] = np.random.randn(self.lora_rank, d_kv).astype(np.float32) * 0.02
                    layer["lora_A_v"], layer["lora_B_v"] = f"lora_A_v_{i}", f"lora_B_v_{i}"
            self.layer_weights.append(layer)

    def _adjust_cache_size(self):
        """Adjust the weight manager's cache size based on the model's total number of tensors."""
        if not isinstance(self.weights, LazyWeightManager):
            return

        # Count all unique weight names assigned to layers
        weight_names = {self.E_name}
        for layer in self.layer_weights:
            for k, v in layer.items():
                if v is None: continue
                if k == "experts":
                    for exp in v:
                        for ek, ev in exp.items():
                            weight_names.add(ev)
                else:
                    weight_names.add(v)

        # Set max_cache_size to cover all tensors plus a small buffer
        new_size = len(weight_names) + 20
        print(f"[INFO] Adjusting LazyWeightManager max_cache_size: {self.weights.max_cache_size} -> {new_size}")
        self.weights.max_cache_size = new_size

    def _validate_loaded_shapes(self):
        c = self.cfg
        errs = []
        if self.E_name and self.E_name in self.weights:
            E = self.weights[self.E_name]
            if E.shape != (c.vocab_size, c.d_model):
                errs.append(f"Embedding {self.E_name}: loaded {E.shape} != config ({c.vocab_size}, {c.d_model})")
        if self.layer_weights and self.layer_weights[0].get("W_q") and self.layer_weights[0]["W_q"] in self.weights:
            Wq = self.weights[self.layer_weights[0]["W_q"]]
            if Wq.shape[0] != c.d_model or Wq.shape[1] != c.d_model:
                errs.append(f"Layer0 W_q: loaded {Wq.shape} != config d_model={c.d_model}")
        if errs: raise ValueError("Checkpoint architecture mismatch.\n  " + "\n  ".join(errs))

    def get_weights_dict(self):
        return {k: np.asarray(self.weights[k]).copy() for k in self.weights.keys()}

    def set_weights_dict(self, d):
        for k, v in d.items(): self.weights[k] = np.asarray(v).copy() if hasattr(v, "copy") else v

    def _quantize_dequant_weight(self, w, bits):
        if bits == 8:
            amax = np.max(np.abs(w))
            if amax == 0 or not np.isfinite(amax): return w
            scale = amax / 127.5
            return np.clip(np.round(w / scale).astype(np.int32), -128, 127).astype(np.float32) * scale
        if bits == 4:
            amax = np.max(np.abs(w))
            if amax == 0 or not np.isfinite(amax): return w
            scale = amax / 7.0
            return np.clip(np.round(w / scale).astype(np.int32), -8, 7).astype(np.float32) * scale
        return w

    def _quantize_dequant_act(self, x, bits):
        if bits == 16:
            return np.clip(x, -65504, 65504).astype(np.float16).astype(np.float32)
        if bits == 8:
            amax = np.max(np.abs(x))
            if amax == 0 or not np.isfinite(amax): return x
            scale = amax / 127.5
            return np.clip(np.round(x / scale).astype(np.int32), -128, 127).astype(np.float32) * scale
        if bits == 4:
            amax = np.max(np.abs(x))
            if amax == 0 or not np.isfinite(amax): return x
            scale = amax / 7.0
            return np.clip(np.round(x / scale).astype(np.int32), -8, 7).astype(np.float32) * scale
        return x

    def get_w(self, name, transpose=False):
        if name is None: return None
        if isinstance(name, np.ndarray): return name.T if transpose else name
        t0 = time.time()
        data = self.weights[name]
        if hasattr(self, '_stats') and 'weight_load' in self._stats:
            self._stats['weight_load'].append(time.time() - t0)

        # Check if we should apply (and cache) simulated quantization
        wb = getattr(self.cfg, "quantization_weight_bits", 0)
        if wb in (4, 8) and not getattr(data, "_is_quantized", False) and "lora" not in str(name).lower():
            data = self._quantize_dequant_weight(data.astype(np.float32), wb)
            # Mark as quantized and store back in weight manager cache to avoid re-computing
            try:
                data._is_quantized = True
                self.weights[name] = data
            except: pass # Some numpy arrays might not allow extra attributes

        return data.T if transpose else data

    @staticmethod
    def _sigmoid(x):
        with np.errstate(over='ignore'):
            return (1.0 / (1.0 + np.exp(-x))).astype(x.dtype, copy=False)

    def rms_norm(self, x, w, eps=1e-6, key=None, out=None):
        t0 = time.time()

        # Optimization: For large tensors, use einsum to compute sum of squares in float64
        # without allocating a full float64 copy of x. This saves memory and is faster.
        # Threshold chosen based on benchmarks (approx 16 * 4096 elements).
        if x.size < 65536:
            # Fast path for small inputs (e.g. single token decode)
            x_fp64 = x.astype(np.float64)
            rms = np.sqrt(np.mean(x_fp64**2, axis=-1, keepdims=True) + eps).astype(x.dtype)
        else:
            # Memory-efficient path for large inputs (e.g. prefill/training)
            # Accumulate sum of squares directly in float64 using einsum
            ss = np.einsum("...d,...d->...", x, x, dtype=np.float64, optimize=True)
            rms = np.sqrt(ss / x.shape[-1] + eps).astype(x.dtype)[..., None]

        # Handle cases where rms might be zero or non-finite
        rms = np.where(np.isfinite(rms) & (rms > 0), rms, eps)
        norm_x = x / rms

        if out is not None:
            np.multiply(norm_x, w, out=out)
            res = out
        else:
            res = norm_x * w

        if hasattr(self, '_stats') and 'norm' in self._stats: self._stats['norm'].append(time.time() - t0)
        if self.training and key is not None: self._activations[key] = {"x": x, "rms": rms, "norm_x": norm_x}
        return res

    def swiglu(self, x, W1, W2, W3, key=None):
        t0 = time.time()

        def proj(mat):
            return np.einsum("btd,df->btf", x, mat, optimize=True)

        # Check total ops for one projection: B * T * D * F
        ops = x.shape[0] * x.shape[1] * x.shape[2] * W1.shape[1]

        if self._should_parallelize(ops):
            futures = [self.executor.submit(proj, W1), self.executor.submit(proj, W2)]
            a = futures[0].result()
            b = futures[1].result()
        else:
            a = proj(W1)
            b = proj(W2)

        # Numerical stable sigmoid to avoid overflow in np.exp
        sig_b = self._sigmoid(b)

        swish_b = b * sig_b
        gated = a * swish_b
        res = np.einsum("btf,fd->btd", gated, W3, optimize=True)
        if hasattr(self, '_stats') and 'ffn_compute' in self._stats:
            self._stats['ffn_compute'].append(time.time() - t0)
        if self.training and key is not None:
            self._activations[key] = {"x": x, "a": a, "b": b, "sig_b": sig_b, "gated": gated}
        return res

    def _log(self, *args, **kwargs):
        if hasattr(self, 'verbose') and not self.verbose: return
        print(*args, **kwargs)

    def apply_rope(self, x, positions):
        B, T, H, D = x.shape

        try:
            # Fast path: use cached RoPE tables
            cos = self.rope_cache["cos"][positions]
            sin = self.rope_cache["sin"][positions]
            # Verify shape match (implicit if index out of bounds raises IndexError, but check len)
            if cos.shape[0] != T:
                 raise IndexError("Position shape mismatch")
        except (AttributeError, IndexError, KeyError, TypeError):
            # Fallback to recomputation
            pos = np.asarray(positions, dtype=np.float64)
            if len(pos) != T:
                if len(pos) > T: pos = pos[:T]
                else: pos = np.concatenate([pos, np.arange(pos[-1]+1, pos[-1]+1 + T - len(pos))])
            inv_freq = 1.0 / (self.cfg.rope_base ** (np.arange(0, D, 2) / D))
            freqs = np.outer(pos, inv_freq)
            cos, sin = np.cos(freqs), np.sin(freqs)

        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def attention(self, x, layer_idx, inference=False):
        t0 = time.time()
        c, p = self.cfg, self.layer_weights[layer_idx]
        B, T = x.shape[0], x.shape[1]

        def compute_proj(name_key):
            t_l = time.time()
            W = self.get_w(p[name_key], transpose=True)
            t_load_weight = time.time() - t_l

            t_p = time.time()
            proj = np.einsum("btd,dh->bth", x, W, optimize=True)

            # Handle LoRA if applicable
            lora_act = None
            if self.use_lora:
                lora_a_key = f"lora_A_{name_key[2:]}" # e.g. W_q -> lora_A_q
                lora_b_key = f"lora_B_{name_key[2:]}"
                if p.get(lora_a_key):
                    scale = self.lora_alpha / self.lora_rank
                    A, Bl = self.get_w(p[lora_a_key]), self.get_w(p[lora_b_key])
                    xA = np.einsum("btd,dr->btr", x, A, optimize=True)
                    proj += scale * np.einsum("btr,rh->bth", xA, Bl, optimize=True)
                    if self.training:
                        lora_act = {"xA": xA, "A": A, "B": Bl, "s": scale}

            t_proj_time = time.time() - t_p
            return name_key, proj, lora_act, t_load_weight, t_proj_time

        # Run Q, K, V projections
        ops = B * T * c.d_model * c.d_model # Approx per projection (W is DxD)

        if self._should_parallelize(ops):
            futures = [self.executor.submit(compute_proj, k) for k in ["W_q", "W_k", "W_v"]]
            results = {}
            t_load_sum = 0
            t_qkv_sum = 0
            for future in as_completed(futures):
                name_key, proj_res, lora_act, t_lw, t_pt = future.result()
                results[name_key] = proj_res
                if lora_act and self.training:
                    self._activations[f"lora_{name_key[2:]}_{layer_idx}"] = lora_act
                t_load_sum += t_lw
                t_qkv_sum += t_pt
        else:
            results = {}
            t_load_sum = 0
            t_qkv_sum = 0
            for k in ["W_q", "W_k", "W_v"]:
                name_key, proj_res, lora_act, t_lw, t_pt = compute_proj(k)
                results[name_key] = proj_res
                if lora_act and self.training:
                    self._activations[f"lora_{name_key[2:]}_{layer_idx}"] = lora_act
                t_load_sum += t_lw
                t_qkv_sum += t_pt

        self._stats['attn_load'].append(t_load_sum / 3.0) # Average or total? Usually stats are for the critical path
        self._stats['attn_qkv'].append(t_qkv_sum / 3.0)

        W_o = self.get_w(p["W_o"], transpose=True)

        Q = results["W_q"].reshape(B, T, c.n_heads, c.d_head)
        K = results["W_k"].reshape(B, T, c.n_kv_heads, c.d_head)
        V = results["W_v"].reshape(B, T, c.n_kv_heads, c.d_head)

        if c.use_mqa or c.use_gqa:
            repeats = c.n_heads // c.n_kv_heads
            K = np.repeat(K, repeats, axis=2)
            V = np.repeat(V, repeats, axis=2)

        t_rope = time.time()
        pos = np.arange(self.cur_pos, self.cur_pos + T) if inference else np.arange(T)
        Q, K = self.apply_rope(Q, pos), self.apply_rope(K, pos)
        self._stats['attn_rope'].append(time.time() - t_rope)

        # Transpose to [Batch, Heads, SeqLen, HeadDim]
        # Convention:
        # B       - Batch size: number of independent sequences processed in parallel
        # H       - Heads: number of attention heads
        # T       - SeqLen: number of tokens in the sequence (time dimension)
        # D_head  - HeadDim: dimension of each individual attention head
        Q_t = Q.transpose(0, 2, 1, 3)
        K_t = K.transpose(0, 2, 1, 3)
        V_t = V.transpose(0, 2, 1, 3)

        if inference:
            # Update or initialize KV cache for iterative decoding
            if self.cache[layer_idx]["K"] is None:
                self.cache[layer_idx]["K"], self.cache[layer_idx]["V"] = K_t, V_t
            else:
                self.cache[layer_idx]["K"] = np.concatenate([self.cache[layer_idx]["K"], K_t], axis=2)
                self.cache[layer_idx]["V"] = np.concatenate([self.cache[layer_idx]["V"], V_t], axis=2)
            K_t, V_t = self.cache[layer_idx]["K"], self.cache[layer_idx]["V"]

        t_score = time.time()
        scores = np.einsum("bhid,bhjd->bhij", Q_t, K_t, optimize=True) / np.sqrt(c.d_head)
        if inference:
            if T == 1: mask = np.ones((1, self.cur_pos + 1))
            else: mask = np.concatenate([np.ones((T, self.cur_pos)), np.tril(np.ones((T, T)))], axis=1)
        else: mask = np.tril(np.ones((T, T)))
        scores = scores * mask - 1e9 * (1 - mask)
        attn_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn_exp / attn_exp.sum(axis=-1, keepdims=True)
        self._stats['attn_score'].append(time.time() - t_score)

        t_out = time.time()
        # Multiply attention weights by values: [B, H, T, T] x [B, H, T, D_head] -> [B, H, T, D_head]
        weighted_v = np.einsum("bhij,bhjd->bhid", attn, V_t, optimize=True)

        # Merge heads back into model dimension
        weighted_v_merged = weighted_v.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)

        # Output projection
        res = np.einsum("btd,dh->bth", weighted_v_merged, W_o, optimize=True)
        self._stats['attn_out'].append(time.time() - t_out)
        self._stats['attn'].append(time.time() - t0)
        if self.training:
            self._activations[f"attn_{layer_idx}"] = {
                "x": x, "Q_t": Q_t, "K_t": K_t, "V_t": V_t,
                "attn": attn, "weighted_v_merged": weighted_v_merged
            }
        return res

    def moe_ffn(self, x, layer_idx):
        t0 = time.time()
        p = self.layer_weights[layer_idx]
        gate_logits = np.einsum("btd,df->btf", x, self.get_w(p["gate"], transpose=True), optimize=True)
        top_indices = np.argsort(gate_logits, axis=-1)[..., -self.cfg.top_k:]
        out = np.zeros_like(x)

        def run_expert(e, b_idx, t_idx):
            # Pre-fetch weights in worker (safetensors handle is thread-safe, but OrderedDict cache is not)
            # Actually, to be safe and avoid OrderedDict contention, we'll pre-fetch in main thread if possible,
            # but here we'll just use the weights.
            W1, W2, W3 = [self.get_w(p["experts"][e][k], transpose=True) for k in ["W1", "W2", "W3"]]
            return e, b_idx, t_idx, self.swiglu(x[b_idx, t_idx][None, :, :], W1, W2, W3)

        futures = []
        for e in range(self.cfg.num_experts):
            b_idx, t_idx, k_idx = np.where(top_indices == e)
            if len(b_idx) == 0: continue
            futures.append(self.executor.submit(run_expert, e, b_idx, t_idx))

        for future in as_completed(futures):
            e, b_idx, t_idx, res_expert = future.result()
            out[b_idx, t_idx] += res_expert[0]

        self._stats['ffn'].append(time.time() - t0)
        return out / self.cfg.top_k

    def forward(self, tokens, inference=False, verbose=True, training=False, max_layers=None):
        if inference and not self.cfg.use_moe:
            # Initialize optimized state on first run
            if self.optimized_params is None:
                self._optimize_for_inference()
            return self._forward_inference(tokens, verbose)

        return self._forward_standard(tokens, inference, verbose, training, max_layers)

    def _optimize_for_inference(self):
        c = self.cfg
        print("[INFO] Optimizing weights for inference (FUSED)...")

        def _sanitize(x):
            # Aggressively clean weights: cast to float32, replace NaNs/Infs with 0
            if x is None: return None
            x = x.astype(np.float32, copy=False)
            return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self.optimized_params = {}
        # Tied weights: Output Head = E.T
        E = _sanitize(self.get_w(self.E_name))
        self.optimized_params['E'] = E
        self.optimized_params['Head'] = E.T

        self.optimized_layers = []
        for i in range(c.n_layers):
            p = self.layer_weights[i]
            layer_w = {}

            def load_linear(key, transpose=True):
                w = self.get_w(p[key], transpose=transpose)
                return _sanitize(np.ascontiguousarray(w))

            wq = load_linear('W_q')
            wk = load_linear('W_k')
            wv = load_linear('W_v')

            if self.use_lora:
                scale = self.lora_alpha / self.lora_rank
                if p.get("lora_A_q"):
                    Aq, Bq = self.get_w(p["lora_A_q"]), self.get_w(p["lora_B_q"])
                    Aq, Bq = _sanitize(Aq), _sanitize(Bq)
                    with np.errstate(all='ignore'):
                        delta = _sanitize(Aq @ Bq)
                    wq = _sanitize(wq + scale * delta)

                if p.get("lora_A_v"):
                    Av, Bv = self.get_w(p["lora_A_v"]), self.get_w(p["lora_B_v"])
                    Av, Bv = _sanitize(Av), _sanitize(Bv)
                    with np.errstate(all='ignore'):
                        delta = _sanitize(Av @ Bv)
                    wv = _sanitize(wv + scale * delta)

            layer_w['W_qkv'] = np.ascontiguousarray(np.concatenate([wq, wk, wv], axis=1))

            layer_w['W_o'] = load_linear('W_o')
            layer_w['norm1'] = _sanitize(self.get_w(p['norm1']))
            layer_w['norm2'] = _sanitize(self.get_w(p['norm2']))

            w1 = load_linear('W1')
            w2 = load_linear('W2')
            layer_w['W_gate_up'] = np.ascontiguousarray(np.concatenate([w1, w2], axis=1))
            layer_w['W3'] = load_linear('W3')

            self.optimized_layers.append(layer_w)

        # Unload original weights to save memory
        if hasattr(self.weights, 'purge_cache'):
            self.weights.purge_cache()

        # Init optimized KV cache
        max_seq = max(c.seq_len, 4096)
        self.kv_cache_opt = np.zeros(
            (c.batch_size, c.n_layers, 2, max_seq, c.n_kv_heads, c.d_head),
            dtype=np.float32
        )

        # Init RoPE
        head_dim = c.d_head
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        self.cos_cached = np.cos(freqs).astype(np.float32)
        self.sin_cached = np.sin(freqs).astype(np.float32)

        # Pre-allocate reusable buffers for single-step decode (L=1)
        B = c.batch_size
        d_q = c.n_heads * c.d_head
        d_kv = c.n_kv_heads * c.d_head

        self.decode_buffers = {
            # QKV projection result: [B, 1, d_q + 2*d_kv]
            "qkv": np.zeros((B, 1, d_q + 2*d_kv), dtype=np.float32),
            # FFN gate_up projection result: [B, 1, 2*c.d_ff]
            "gate_up": np.zeros((B, 1, 2*c.d_ff), dtype=np.float32),
            # Shared buffer for outputs (Attn out, FFN down, Norm): [B, 1, c.d_model]
            "out": np.zeros((B, 1, c.d_model), dtype=np.float32),
            # Buffer for Norm result: [B, 1, c.d_model]
            "h": np.zeros((B, 1, c.d_model), dtype=np.float32)
        }

    def _forward_inference(self, tokens, verbose=True):
        if not hasattr(self, '_stats'): self._init_stats()
        # Helper to track stats similar to standard forward
        t_stage_start = time.time()

        B, L = tokens.shape
        c = self.cfg
        start_pos = self.cur_pos

        if B > c.batch_size:
            raise ValueError(f"Batch size {B} exceeds allocated {c.batch_size}")

        t_emb = time.time()
        x = np.take(self.optimized_params['E'], tokens, axis=0) # [B, L, D]
        self._stats['embedding'].append(time.time() - t_emb)

        d_q = c.n_heads * c.d_head
        d_kv = c.n_kv_heads * c.d_head
        d_gate = c.d_ff

        cos_T = self.cos_cached[start_pos : start_pos+L]
        sin_T = self.sin_cached[start_pos : start_pos+L]

        if L == 1:
            cos_T = cos_T[None, None, :, :]
            sin_T = sin_T[None, None, :, :]
        else:
            cos_T = cos_T[None, :, None, :]
            sin_T = sin_T[None, :, None, :]

        mask = None
        if L > 1:
            mask = np.triu(np.ones((L, L), dtype=np.float32), k=1) * -1e9

        # Use pre-allocated buffers if applicable
        bufs = None
        if L == 1 and B == c.batch_size and hasattr(self, 'decode_buffers'):
            bufs = self.decode_buffers

        t_layers_total = 0

        for i, l in enumerate(self.optimized_layers):
            t_l_start = time.time()

            t_norm = time.time()
            # Norm 1
            if bufs:
                h = self.rms_norm(x, l['norm1'], out=bufs['h'])
            else:
                h = self.rms_norm(x, l['norm1'])
            # Ensure stability of activations
            h = np.nan_to_num(h, copy=False, nan=0.0, posinf=65504.0, neginf=-65504.0)
            h = np.ascontiguousarray(h)
            self._stats['norm_step'].append(time.time() - t_norm)

            t_qkv = time.time()
            # QKV
            if bufs:
                qkv = tiled_matmul(h, l['W_qkv'], executor=self.executor, backend=c.backend, out=bufs['qkv'])
            else:
                qkv = tiled_matmul(h, l['W_qkv'], executor=self.executor, backend=c.backend)
            self._stats['attn_qkv_step'].append(time.time() - t_qkv)

            q = qkv[..., :d_q]
            k = qkv[..., d_q:d_q+d_kv]
            v = qkv[..., d_q+d_kv:]

            q = q.reshape(B, L, c.n_heads, c.d_head)
            k = k.reshape(B, L, c.n_kv_heads, c.d_head)
            v = v.reshape(B, L, c.n_kv_heads, c.d_head)

            t_rope = time.time()
            # RoPE
            q_r, q_i = q[..., 0::2], q[..., 1::2]
            k_r, k_i = k[..., 0::2], k[..., 1::2]

            c_half = cos_T
            s_half = sin_T

            q_r_new = q_r * c_half - q_i * s_half
            q_i_new = q_r * s_half + q_i * c_half
            q[..., 0::2] = q_r_new
            q[..., 1::2] = q_i_new

            k_r_new = k_r * c_half - k_i * s_half
            k_i_new = k_r * s_half + k_i * c_half
            self._stats['attn_rope_step'].append(time.time() - t_rope)

            # Update Cache
            self.kv_cache_opt[:B, i, 0, start_pos:start_pos+L, :, :] = k
            self.kv_cache_opt[:B, i, 1, start_pos:start_pos+L, :, :] = v

            k_cache = self.kv_cache_opt[:B, i, 0, :start_pos+L, :, :]
            v_cache = self.kv_cache_opt[:B, i, 1, :start_pos+L, :, :]

            t_attn_score = time.time()
            # Attention
            if c.n_heads != c.n_kv_heads:
                n_rep = c.n_heads // c.n_kv_heads
                q_reshaped = q.reshape(B, L, c.n_kv_heads, n_rep, c.d_head).transpose(0, 2, 1, 3, 4)

                k_t = k_cache.transpose(0, 2, 3, 1)
                k_t = k_t[:, :, None, :, :]

                scores = tiled_matmul(q_reshaped, k_t, executor=self.executor, backend=c.backend) / np.sqrt(c.d_head)

                if mask is not None:
                    scores[..., :, :, start_pos:] += mask.reshape(L, 1, L)

                max_score = np.max(scores, axis=-1, keepdims=True)
                exp_score = np.exp(scores - max_score)
                attn = exp_score / np.sum(exp_score, axis=-1, keepdims=True)

                v_t = v_cache.transpose(0, 2, 1, 3)
                v_t = v_t[:, :, None, :, :]

                out = tiled_matmul(attn, v_t, executor=self.executor, backend=c.backend)
                out = out.transpose(0, 2, 1, 3, 4).reshape(B, L, c.d_model)

            else:
                q_t = q.transpose(0, 2, 1, 3)
                k_t = k_cache.transpose(0, 2, 3, 1)

                scores = tiled_matmul(q_t, k_t, executor=self.executor, backend=c.backend) / np.sqrt(c.d_head)

                if mask is not None:
                    scores[..., :, start_pos:] += mask

                max_score = np.max(scores, axis=-1, keepdims=True)
                exp_score = np.exp(scores - max_score)
                attn = exp_score / np.sum(exp_score, axis=-1, keepdims=True)

                v_t = v_cache.transpose(0, 2, 1, 3)
                out = tiled_matmul(attn, v_t, executor=self.executor, backend=c.backend)
                out = out.transpose(0, 2, 1, 3).reshape(B, L, c.d_model)
            self._stats['attn_score_step'].append(time.time() - t_attn_score)

            t_out = time.time()
            if bufs:
                h = tiled_matmul(out, l['W_o'], backend=c.backend, out=bufs['out'])
            else:
                h = tiled_matmul(out, l['W_o'], backend=c.backend)
            x = x + h
            self._stats['attn_out_step'].append(time.time() - t_out)

            t_ffn_s = time.time()
            # FFN
            if bufs:
                h = self.rms_norm(x, l['norm2'], out=bufs['h'])
            else:
                h = self.rms_norm(x, l['norm2'])
            h = np.nan_to_num(h, copy=False, nan=0.0, posinf=65504.0, neginf=-65504.0)
            h = np.ascontiguousarray(h)

            if bufs:
                gate_up = tiled_matmul(h, l['W_gate_up'], backend=c.backend, out=bufs['gate_up'])
            else:
                gate_up = tiled_matmul(h, l['W_gate_up'], backend=c.backend)
            gate = gate_up[..., :d_gate]
            up = gate_up[..., d_gate:]

            gate = gate * self._sigmoid(gate) * up

            if bufs:
                down = tiled_matmul(gate, l['W3'], backend=c.backend, out=bufs['out'])
            else:
                down = tiled_matmul(gate, l['W3'], backend=c.backend)
            x = x + down
            self._stats['ffn_compute_step'].append(time.time() - t_ffn_s)

            # Aggregate layer time
            self._stats['layers_step'].append(time.time() - t_l_start)
            t_layers_total += (time.time() - t_l_start)

        t_head = time.time()
        logits = tiled_matmul(x, self.optimized_params['Head'], backend=c.backend)
        self._stats['output_head'].append(time.time() - t_head)

        # Stats for stage
        if L > 1:
            self._stats['prefill'].append(time.time() - t_stage_start)
        else:
            self._stats['decode'].append(time.time() - t_stage_start)

        if verbose:
             self._print_stats_table()

        # Only advance position if not speculative verification (verification advances manually or logic handles it)
        # For standard generation, we advance.
        # But this function is used by generate() loop which expects manual advance if called in chunks?
        # Actually generate() calls this per step.
        # Let's side-effect update cur_pos for simplicity in standard inference
        self.cur_pos += L
        return logits

    def _forward_standard(self, tokens, inference=False, verbose=True, training=False, max_layers=None):
        """
        Forward pass of the Transformer using pure einsum notation.

        HIGH-LEVEL FLOW:
        ================
        1. Embedding Lookup: tokens -> x via E[tokens]
        2. For each layer (N layers):
           a. Pre-Norm 1: x -> x_norm1 via RMSNorm
           b. Multi-Head Attention (with GQA/MQA support):
              - QKV Projections: x_norm1 @ W_q/W_k/W_v -> Q, K, V
              - RoPE: Apply rotary positional embeddings to Q, K
              - Attention Scores: Q @ K^T / sqrt(d_head) -> scores
              - Causal Masking: scores * mask
              - Softmax: exp(scores) / sum(exp(scores)) -> attn
              - Weighted Aggregation: attn @ V -> weighted_v
              - Output Projection: weighted_v @ W_o -> attn_out
           c. Residual Connection: x = x + attn_out
           d. Pre-Norm 2: x -> x_norm2 via RMSNorm
           e. Feed-Forward Network (SwiGLU):
              - Gate: x_norm2 @ W1 -> a
              - Up: x_norm2 @ W2 -> b
              - Activation: a * swish(b) -> gated
              - Down: gated @ W3 -> ffn_out
           f. Residual Connection: x = x + ffn_out
        3. Output Head: x @ E^T -> logits

        KEY EINSUM OPERATIONS:
        ======================
        Embedding:        E[tokens] -> (B, T, D)
        QKV Projection:   einsum('btd,dh->bth', x, W_q) -> Q  [B=batch, T=seq, D=d_model, H=n_heads*d_head]
        Attention Scores: einsum('bhid,bhjd->bhij', Q_t, K_t) -> scores  [h=heads, i,j=positions]
        Attention Output: einsum('bhij,bhjd->bhid', attn, V_t) -> weighted_v
        Output Proj:      einsum('btd,dh->bth', weighted_v_merged, W_o) -> attn_out
        SwiGLU Gate:      einsum('btd,df->btf', x, W1) -> a  [f=d_ff]
        SwiGLU Down:      einsum('btf,fd->btd', gated, W3) -> ffn_out
        Output Head:      einsum('btd,vd->btv', x, E) -> logits  [v=vocab_size]
        """
        self.verbose, self.training = verbose, training
        if not hasattr(self, '_stats'): self._init_stats()

        # Record start lengths of layered stats to aggregate them later
        layered_keys = ['layers', 'norm', 'weight_load', 'attn', 'attn_load', 'attn_qkv',
                        'attn_rope', 'attn_score', 'attn_out', 'ffn', 'ffn_load', 'ffn_compute']
        stat_starts = {k: len(self._stats[k]) for k in layered_keys}

        if training: self._activations, self._grads = {"tokens": tokens}, {}
        is_decode = inference and any(c["K"] is not None for c in self.cache)
        if inference and not is_decode: self.cur_pos = 0
        t_stage_start = time.time()

        c = self.cfg
        if self.verbose and not is_decode:
            self._log("\n" + "="*50)
            self._log(f"MODEL PROFILING: {c.arch.upper()} {'(TRAINING)' if training else ''}")
            self._log(f"Input Dimension:  {tokens.shape}")
            self._log(f"Hidden Dimension: {c.d_model}")
            self._log(f"Output Dimension: {c.vocab_size}")
            self._log(f"Layers:           {c.n_layers}")
            self._log(f"Max Context Len:  {c.seq_len}")
            self._log("="*50)
            self._log("[INFO] Starting forward pass...")

        t_embed = time.time()
        E = self.get_w(self.E_name)
        x = E[tokens]
        if 'embedding' not in self._stats: self._stats['embedding'] = []
        self._stats['embedding'].append(time.time() - t_embed)
        ab = getattr(self.cfg, "quantization_activation_bits", 0)
        if ab in (4, 8, 16): x = self._quantize_dequant_act(x.astype(np.float32), ab)
        if training: self._activations["embedding"] = x
        
        n_layers = max_layers if max_layers is not None else self.cfg.n_layers

        # Track aggregate time for all layers in this step
        t_layers_step = 0

        for i in range(n_layers):
            t_layer_start = time.time()
            p = self.layer_weights[i]
            x_norm = self.rms_norm(x, self.get_w(p["norm1"]), key=f"norm1_{i}")
            if ab in (4, 8, 16): x_norm = self._quantize_dequant_act(x_norm.astype(np.float32), ab)
            x = x + self.attention(x_norm, i, inference)
            x_norm = self.rms_norm(x, self.get_w(p["norm2"]), key=f"norm2_{i}")
            if ab in (4, 8, 16): x_norm = self._quantize_dequant_act(x_norm.astype(np.float32), ab)
            if self.cfg.use_moe: x = x + self.moe_ffn(x_norm, i)
            else:
                t_ffn_start = time.time()
                t_fload = time.time()
                W1, W2, W3 = [self.get_w(p[k], transpose=True) for k in ["W1", "W2", "W3"]]
                self._stats['ffn_load'].append(time.time() - t_fload)
                ffn_out = self.swiglu(x_norm, W1, W2, W3, key=f"ffn_{i}")
                x = x + ffn_out
                self._stats['ffn'].append(time.time() - t_ffn_start)

            t_layer = time.time() - t_layer_start
            self._stats['layers'].append(t_layer)
            t_layers_step += t_layer

        self._stats['layers_step'].append(t_layers_step)

        # Aggregate layered stats for this forward pass
        for k in layered_keys:
            if k == 'layers': continue # handled above as layers_step
            self._stats[k + '_step'].append(sum(self._stats[k][stat_starts[k]:]))

        if inference: self.cur_pos += tokens.shape[1]
        if not is_decode and inference: self._stats['prefill'].append(time.time() - t_stage_start)
        elif is_decode: self._stats['decode'].append(time.time() - t_stage_start)

        if training:
            self._activations["final_x"] = x
            # Calculate activation memory footprint correctly
            def _sum_bytes(d):
                b = 0
                for v in d.values():
                    if isinstance(v, np.ndarray): b += v.nbytes
                    elif isinstance(v, dict): b += _sum_bytes(v)
                return b
            w_mem = self.weights.get_cache_memory_mb() if hasattr(self.weights, 'get_cache_memory_mb') else 0
            self._memory_stats['fwd_activation_mb'] = (_sum_bytes(self._activations) / (1024 * 1024)) + w_mem

        t_head = time.time()
        res = np.einsum("btd,vd->btv", x, E, optimize=True)
        if 'output_head' not in self._stats: self._stats['output_head'] = []
        self._stats['output_head'].append(time.time() - t_head)
        if self.verbose and not self.training and not is_decode: self._print_stats_table()
        return res

    def backward(self, d_logits):
        """
        Backward pass computing all gradients using pure einsum notation.

        HIGH-LEVEL FLOW:
        ================
        1. Output Head Backward:
           - Compute dx from d_logits and E
           - Compute dE (embedding gradient) from d_logits and final_x

        2. For each layer (N-1 to 0, reverse order):
           a. FFN Backward (SwiGLU):
              - Backprop through W3 projection
              - Backprop through SwiGLU activation (gate * swish derivative)
              - Compute dW3, dW1, dW2 gradients
              - Compute dx contribution from FFN
           b. Norm2 Backward:
              - Compute d_norm2 (RMSNorm weight gradient)
              - Backprop through RMSNorm (mean-centered gradient normalization)
              - Accumulate dx from FFN residual path
           c. Attention Backward:
              - Backprop through W_o projection
              - Backprop through attention output aggregation
              - Backprop through Softmax (Jacobian: attn * (d_attn - sum(d_attn * attn)))
              - Backprop through attention scores (QK^T / sqrt(d_head))
              - Handle GQA: sum gradients across query groups for K and V
              - Backprop through RoPE (inverse rotation via transpose)
              - Backprop through QKV projections
              - Compute dW_q, dW_k, dW_v, dW_o gradients
              - Compute dx contribution from attention
           d. Norm1 Backward:
              - Compute d_norm1 gradient
              - Backprop through RMSNorm
              - Accumulate dx from attention residual path

        3. Embedding Backward:
           - Accumulate dE from token lookup indices

        KEY EINSUM OPERATIONS (BACKWARD):
        =================================
        Output Head:
          dx = einsum('btv,vd->btd', d_logits, E)
          dE = einsum('btv,btd->vd', d_logits, final_x)

        FFN (SwiGLU):
          dW3 = einsum('btf,btd->fd', gated, dffn_out)
          dgated = einsum('btd,fd->btf', dffn_out, W3)
          dW1 = einsum('btd,btf->df', x, da)  [da = dgated * swish(b)]
          dW2 = einsum('btd,btf->df', x, db)  [db = dgated * a * swish'(b)]
          dx += einsum('btf,df->btd', da, W1) + einsum('btf,df->btd', db, W2)

        RMSNorm:
          d_norm = einsum('btd,btd->d', dx_norm, norm_x)  [norm_x = x / rms]
          dx = (w / rms) * (dx_norm - mean(dx_norm * norm_x) * norm_x)

        Attention:
          dW_o = einsum('btd,bth->dh', weighted_v_merged, dattn_out)
          dweighted_merged = einsum('bth,dh->btd', dattn_out, W_o)
          dattn = einsum('bhid,bhjd->bhij', dweighted, V_t)
          dscores = attn * (dattn - sum(dattn * attn, axis=-1))
          dQ_t = einsum('bhij,bhjd->bhid', dscores, K_t)
          dK_t_full = einsum('bhji,bhjd->bhid', dscores, Q_t)
          dV_t_full = einsum('bhij,bhid->bhjd', attn, dweighted)
          # GQA: sum across groups
          dK_t = dK_t_full.reshape(...).sum(axis=group_dim)
          dV_t = dV_t_full.reshape(...).sum(axis=group_dim)
          # RoPE inverse
          dQ, dK = rope_backward(dQ_t), rope_backward(dK_t)
          dW_q = einsum('btd,bth->dh', x, dQ.reshape(...))
          dW_k = einsum('btd,bth->dh', x, dK.reshape(...))
          dW_v = einsum('btd,bth->dh', x, dV.reshape(...))
          dx += einsum('bth,dh->btd', dQ, W_q) + einsum('bth,dh->btd', dK, W_k) + einsum('bth,dh->btd', dV, W_v)

        GRADIENTS COMPUTED:
        ===================
        - E: Embedding weight gradient (vocab_size, d_model)
        - Per-layer:
          * W_q, W_k, W_v, W_o: Attention weight gradients (d_model, d_model)
          * norm1, norm2: RMSNorm weight gradients (d_model,)
          * W1, W2, W3: FFN weight gradients (d_model, d_ff) or (d_ff, d_model)
        """
        t_total = time.time()

        # Record start lengths of layered stats to aggregate them later
        layered_keys = ['bw_layers', 'bw_norm', 'bw_attn', 'bw_ffn', 'bw_attn_out', 'bw_attn_score', 'bw_attn_qkv', 'bw_ffn_compute']
        stat_starts = {k: len(self._stats[k]) for k in layered_keys}

        c = self.cfg
        E = self.get_w(self.E_name)
        final_x = self._activations["final_x"]

        t_head = time.time()
        # Gradient with respect to the final layer activations
        # [B, T, V] x [V, D] -> [B, T, D]
        dx = np.einsum("btv,vd->btd", d_logits, E, optimize=True)

        if not self.use_lora:
            # Gradient with respect to the output embedding (tied with input embedding)
            self._grads["E"] = np.einsum("btv,btd->vd", d_logits, final_x, optimize=True)
        self._stats['bw_head'].append(time.time() - t_head)

        for i in reversed(range(c.n_layers)):
            t_layer_start = time.time()
            p = self.layer_weights[i]
            if not c.use_moe:
                t_ffn = time.time()
                t_ffn_compute = time.time()
                act_ffn, dffn_out = self._activations[f"ffn_{i}"], dx
                W1, W2, W3 = [self.get_w(p[k], transpose=True) for k in ["W1", "W2", "W3"]]
                if not self.use_lora: self._grads[f"W3_{i}"] = np.einsum("btf,btd->fd", act_ffn["gated"], dffn_out, optimize=True)
                dgated = np.einsum("btd,fd->btf", dffn_out, W3, optimize=True)
                da = dgated * (act_ffn["b"] * act_ffn["sig_b"])
                db = dgated * act_ffn["a"] * (act_ffn["sig_b"] * (1 + act_ffn["b"] * (1 - act_ffn["sig_b"])))
                if not self.use_lora:
                    self._grads[f"W1_{i}"], self._grads[f"W2_{i}"] = np.einsum("btd,btf->df", act_ffn["x"], da, optimize=True), np.einsum("btd,btf->df", act_ffn["x"], db, optimize=True)
                dx = dx + np.einsum("btf,df->btd", da, W1, optimize=True) + np.einsum("btf,df->btd", db, W2, optimize=True)
                self._stats['bw_ffn_compute'].append(time.time() - t_ffn_compute)

                t_norm2 = time.time()
                act_norm2, dx_norm2 = self._activations[f"norm2_{i}"], dx
                norm2_w = self.get_w(p["norm2"])
                if not self.use_lora: self._grads[f"norm2_{i}"] = np.einsum("btd,btd->d", dx_norm2, act_norm2["norm_x"], optimize=True)
                term1 = dx_norm2 * norm2_w
                dx = dx + (term1 - np.mean(term1 * act_norm2["norm_x"], axis=-1, keepdims=True) * act_norm2["norm_x"]) / act_norm2["rms"]
                self._stats['bw_norm'].append(time.time() - t_norm2)
                self._stats['bw_ffn'].append(time.time() - t_ffn)

            t_attn = time.time()
            act_attn, dattn_out = self._activations[f"attn_{i}"], dx
            W_o = self.get_w(p["W_o"], transpose=True)

            t_bw_attn_out = time.time()
            if not self.use_lora: self._grads[f"W_o_{i}"] = np.einsum("btd,bth->dh", act_attn["weighted_v_merged"], dattn_out, optimize=True)
            # Backward through Output Projection and Merge
            d_weighted_v_merged = np.einsum("bth,dh->btd", dattn_out, W_o, optimize=True)
            d_weighted_v = d_weighted_v_merged.reshape(dx.shape[0], dx.shape[1], c.n_heads, c.d_head).transpose(0, 2, 1, 3)
            self._stats['bw_attn_out'].append(time.time() - t_bw_attn_out)

            t_bw_attn_score = time.time()
            # Backward through Attention Softmax and Weighted Sum
            # dattn: [B, H, T, T]
            dattn = np.einsum("bhid,bhjd->bhij", d_weighted_v, act_attn["V_t"], optimize=True)

            # Backward through Softmax
            dscores = act_attn["attn"] * (dattn - np.sum(dattn * act_attn["attn"], axis=-1, keepdims=True)) / np.sqrt(c.d_head)
            self._stats['bw_attn_score'].append(time.time() - t_bw_attn_score)

            t_bw_attn_qkv = time.time()
            # Gradients for Q, K, V projections
            dQ_t = np.einsum("bhij,bhjd->bhid", dscores, act_attn["K_t"], optimize=True)
            dK_t_f = np.einsum("bhji,bhjd->bhid", dscores, act_attn["Q_t"], optimize=True)
            dV_t_f = np.einsum("bhij,bhid->bhjd", act_attn["attn"], d_weighted_v, optimize=True)
            groups = c.n_heads // c.n_kv_heads
            dQ, dK = dQ_t.transpose(0, 2, 1, 3), dK_t_f.reshape(dx.shape[0], c.n_kv_heads, groups, -1, c.d_head).sum(2).transpose(0, 2, 1, 3)
            dV = dV_t_f.reshape(dx.shape[0], c.n_kv_heads, groups, -1, c.d_head).sum(2).transpose(0, 2, 1, 3)
            # Backward through RoPE
            dQ = self._rope_backward(dQ, np.arange(dQ.shape[1]))
            dK = self._rope_backward(dK, np.arange(dK.shape[1]))

            # Reshape gradients and apply clipping for numerical stability
            dQf = np.clip(dQ.reshape(dx.shape[0], dx.shape[1], -1), -1.0, 1.0)
            dKf = np.clip(dK.reshape(dx.shape[0], dx.shape[1], -1), -1.0, 1.0)
            dVf = np.clip(dV.reshape(dx.shape[0], dx.shape[1], -1), -1.0, 1.0)
            dQ_B = dV_B = None
            if self.use_lora:
                if f"lora_q_{i}" in self._activations:
                    al = self._activations[f"lora_q_{i}"]
                    self._grads[f"lora_B_q_{i}"], dQ_B = al["s"] * np.einsum("btr,bth->rh", al["xA"], dQf, optimize=True), np.einsum("bth,rh->btr", dQf, al["B"], optimize=True)
                    self._grads[f"lora_A_q_{i}"] = al["s"] * np.einsum("btd,btr->dr", act_attn["x"], dQ_B, optimize=True)
                if f"lora_v_{i}" in self._activations:
                    al = self._activations[f"lora_v_{i}"]
                    self._grads[f"lora_B_v_{i}"], dV_B = al["s"] * np.einsum("btr,bth->rh", al["xA"], dVf, optimize=True), np.einsum("bth,rh->btr", dVf, al["B"], optimize=True)
                    self._grads[f"lora_A_v_{i}"] = al["s"] * np.einsum("btd,btr->dr", act_attn["x"], dV_B, optimize=True)
            if not self.use_lora:
                def compute_grad_w(t):
                    return np.einsum("btd,bth->dh", act_attn["x"], t, optimize=True)

                grad_futures = [self.executor.submit(compute_grad_w, t) for t in [dQf, dKf, dVf]]
                self._grads[f"W_q_{i}"] = grad_futures[0].result()
                self._grads[f"W_k_{i}"] = grad_futures[1].result()
                self._grads[f"W_v_{i}"] = grad_futures[2].result()

            W_q, W_k, W_v = [self.get_w(p[k], transpose=True) for k in ["W_q", "W_k", "W_v"]]

            def compute_dx_term(t, W):
                return np.einsum("bth,dh->btd", t, W, optimize=True)

            dx_futures = [self.executor.submit(compute_dx_term, dQf, W_q),
                          self.executor.submit(compute_dx_term, dKf, W_k),
                          self.executor.submit(compute_dx_term, dVf, W_v)]
            dx = dx + dx_futures[0].result() + dx_futures[1].result() + dx_futures[2].result()
            if self.use_lora:
                if dQ_B is not None: dx += self._activations[f"lora_q_{i}"]["s"] * np.einsum("btr,dr->btd", dQ_B, self._activations[f"lora_q_{i}"]["A"], optimize=True)
                if dV_B is not None: dx += self._activations[f"lora_v_{i}"]["s"] * np.einsum("btr,dr->btd", dV_B, self._activations[f"lora_v_{i}"]["A"], optimize=True)
            self._stats['bw_attn_qkv'].append(time.time() - t_bw_attn_qkv)

            t_norm1 = time.time()
            act_norm1, dx_norm1 = self._activations[f"norm1_{i}"], dx
            norm1_w = self.get_w(p["norm1"])
            if not self.use_lora: self._grads[f"norm1_{i}"] = np.einsum("btd,btd->d", dx_norm1, act_norm1["norm_x"], optimize=True)
            term1 = dx_norm1 * norm1_w
            dx = dx + (term1 - np.mean(term1 * act_norm1["norm_x"], axis=-1, keepdims=True) * act_norm1["norm_x"]) / act_norm1["rms"]
            self._stats['bw_norm'].append(time.time() - t_norm1)
            self._stats['bw_attn'].append(time.time() - t_attn)
            self._stats['bw_layers'].append(time.time() - t_layer_start)

        t_embed = time.time()
        if not self.use_lora:
            # Gradient for the input embedding (accumulated via add.at for multi-token indices)
            np.add.at(self._grads["E"], self._activations["tokens"], dx)
        self._stats['bw_embed'].append(time.time() - t_embed)

        # Aggregate layered stats for this backward pass
        for k in layered_keys:
            self._stats[k + '_step'].append(sum(self._stats[k][stat_starts[k]:]))

        # Calculate gradient memory footprint
        total_grad_bytes = sum(g.nbytes for g in self._grads.values() if isinstance(g, np.ndarray))
        w_mem = self.weights.get_cache_memory_mb() if hasattr(self.weights, 'get_cache_memory_mb') else 0
        self._memory_stats['grad_total_mb'] = (total_grad_bytes / (1024 * 1024)) + w_mem

        self._stats['bw_total'].append(time.time() - t_total)
        return self._grads

    def generate(self, tokens):
        if getattr(self.cfg, 'speculative', False): return self.speculative_generate(tokens)
        self.clear_cache()
        B = tokens.shape[0]

        t0 = time.time()
        # Prefill
        # logits = self.forward(tokens, inference=True, verbose=True)
        # Using forward with inference=True handles optimization internally
        logits = self.forward(tokens, inference=True, verbose=True)

        next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
        ttft = time.time() - t0

        # generated is a list of arrays, each of shape (B,)
        generated = [next_token]

        print(f"\n Decoding Stage")
        print(f"=================")
        # Print stream for the first batch element
        print(f"[STREAM] {self.detokenize([next_token[0]])[0]}", end="", flush=True)

        t_decode_start = time.time()
        for i in range(self.cfg.max_new_tokens - 1):
            # Input shape: (B, 1)
            logits = self.forward(generated[-1][:, None], inference=True, verbose=False)
            next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
            generated.append(next_token)
            print(self.detokenize([next_token[0]])[0], end="", flush=True)

        t_decode_end = time.time()
        duration = t_decode_end - t_decode_start
        tokens_sec = (len(generated) - 1) / duration if duration > 0 else 0

        print(f"\nTime to first Token ({self.detokenize([generated[0][0]])[0]}) : {ttft:.4f}s")
        print(f"Output tokens/sec : {tokens_sec:.2f}")
        print()
        return generated

    def _get_draft_model(self):
        if self._draft_model is None or self._draft_model.cfg.n_layers != self.cfg.draft_layers:
            # Initialize draft model using the first N layers of the target model
            draft_cfg = copy.deepcopy(self.cfg)
            draft_cfg.n_layers = self.cfg.draft_layers
            # Disable speculative in draft to avoid infinite recursion
            draft_cfg.speculative = False
            self._draft_model = EinsumTransformer(draft_cfg, _from_weights=self.weights,
                                            _E_name=self.E_name,
                                            _layer_weights=self.layer_weights[:self.cfg.draft_layers])
            # Share the executor to avoid creating too many threads
            self._draft_model.executor = self.executor
        return self._draft_model

    def speculative_generate(self, tokens, k=4):
        """
        Implementation of speculative decoding.
        A smaller 'draft' model (first few layers of the current model) predicts k tokens,
        which are then verified in parallel by the full 'target' model.
        """
        print(f"[INFO] Starting speculative decoding (k={k}, draft_layers={self.cfg.draft_layers})...")
        self.clear_cache()
        draft_model = self._get_draft_model()
        draft_model.clear_cache()
        B = tokens.shape[0]
        
        t0 = time.time()
        # 1. Prefill: Process initial prompt tokens in target model
        logits = self.forward(tokens, inference=True, verbose=True)

        # 2. Seed draft model KV cache from target model to avoid redundant prefill
        # NOTE: Optimized KV cache is separate. If optimization is on, we need to sync optimized caches.
        if self.kv_cache_opt is not None:
            # Ensure draft model is also optimized
            if draft_model.kv_cache_opt is None:
                draft_model._optimize_for_inference()

            for i in range(self.cfg.draft_layers):
                draft_model.kv_cache_opt[:, i] = self.kv_cache_opt[:, i]
        else:
            for i in range(self.cfg.draft_layers):
                draft_model.cache[i]["K"] = self.cache[i]["K"]
                draft_model.cache[i]["V"] = self.cache[i]["V"]

        draft_model.cur_pos = self.cur_pos
        
        # 2. Sample the first token after the prompt
        next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
        ttft = time.time() - t0

        generated = [next_token]
        print(f"\n Decoding Stage")
        print(f"=================")
        print(f"[STREAM] {self.detokenize([next_token[0]])[0]}", end="", flush=True)

        t_decode_start = time.time()
        # 3. Speculative loop: Iterate until max_new_tokens is reached
        while len(generated) < self.cfg.max_new_tokens:
            # a. Draft predicts k speculative tokens sequentially
            d_tokens = []
            draft_input = generated[-1][:, None]
            for _ in range(k):
                d_logits = draft_model.forward(draft_input, inference=True, verbose=False)
                d_tok = self.sample_top_k(d_logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
                d_tokens.append(d_tok)
                draft_input = d_tok[:, None]

            # b. Target verifies all k draft tokens + current token in a single parallel pass.
            # v_input[j] will predict the token for d_tokens[j].
            # v_input[k] (the last draft token) will predict a "bonus" token.
            v_input = np.stack([generated[-1]] + d_tokens, axis=1) # Shape (B, k+1)
            v_logits = self.forward(v_input, inference=True, verbose=False)

            # c. Verify draft tokens against target model predictions
            n_accepted = 0
            next_toks = None
            for j in range(k):
                # Sample the target model's prediction for the token at d_tokens[j] position
                target_toks = self.sample_top_k(v_logits[:, j:j+1, :], k=self.cfg.top_k_sample, temperature=self.cfg.temperature)

                # In batched mode, we only accept if ALL batches matched for this step
                # to keep KV caches aligned across the batch.
                if np.array_equal(target_toks, d_tokens[j]):
                    # Draft was correct!
                    generated.append(d_tokens[j])
                    print(self.detokenize([d_tokens[j][0]])[0], end="", flush=True)
                    n_accepted += 1
                else:
                    # Mismatch. Accept the target's correction and stop verification.
                    generated.append(target_toks)
                    print(f"[{self.detokenize([target_toks[0]])[0]}]", end="", flush=True)
                    next_toks = target_toks
                    break

            # If all were accepted, we get a bonus token from the last logit position
            if n_accepted == k:
                next_toks = self.sample_top_k(v_logits[:, k:k+1, :], k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
                generated.append(next_toks)
                print(self.detokenize([next_toks[0]])[0], end="", flush=True)

            # d. Rollback target model KV cache
            n_to_keep = n_accepted + 1
            n_target_rollback = (k + 1) - n_to_keep

            if n_target_rollback > 0:
                self.cur_pos -= n_target_rollback
                # Support Optimized Cache rollback
                # Optimized cache is just a circular buffer or linear buffer.
                # Since we write linearly, we just decrement cur_pos.
                # No data clearing needed as it will be overwritten.

                # Standard Cache rollback (if used, but here we use optimized likely)
                if self.kv_cache_opt is None:
                    for layer in self.cache:
                        if layer["K"] is not None:
                            layer["K"] = layer["K"][:, :, :-n_target_rollback, :]
                            layer["V"] = layer["V"][:, :, :-n_target_rollback, :]

            # e. Synchronize draft model state with target model
            draft_model.cur_pos = self.cur_pos
            if self.kv_cache_opt is not None:
                # Optimized sync
                # Only need to sync the valid part up to cur_pos
                # Copy from (cur_pos - n_to_keep) to cur_pos.
                start_sync = max(0, self.cur_pos - n_to_keep)
                # Draft cache has fewer layers (draft_layers), self cache has n_layers
                n_d = self.cfg.draft_layers
                # Slice ONLY the draft layers from the main cache
                draft_model.kv_cache_opt[:, :n_d, :, start_sync:self.cur_pos, :, :] = self.kv_cache_opt[:, :n_d, :, start_sync:self.cur_pos, :, :]
            else:
                for l in range(self.cfg.draft_layers):
                    draft_model.cache[l]["K"] = self.cache[l]["K"]
                    draft_model.cache[l]["V"] = self.cache[l]["V"]

        t_decode_end = time.time()
        duration = t_decode_end - t_decode_start
        tokens_sec = (len(generated) - 1) / duration if duration > 0 else 0

        print(f"\nTime to first Token ({self.detokenize([generated[0][0]])[0]}) : {ttft:.4f}s")
        print(f"Output tokens/sec : {tokens_sec:.2f}")
        print()
        return generated

    def _init_stats(self):
        self._stats = {k: [] for k in [
            'attn', 'ffn', 'layers', 'norm', 'weight_load', 'attn_load', 'attn_qkv', 'attn_rope', 'attn_score', 'attn_out', 'ffn_load', 'ffn_compute',
            'prefill', 'decode', 'embedding', 'output_head',
            'layers_step', 'norm_step', 'weight_load_step', 'attn_step', 'attn_load_step', 'attn_qkv_step', 'attn_rope_step', 'attn_score_step', 'attn_out_step', 'ffn_step', 'ffn_load_step', 'ffn_compute_step',
            'bw_total', 'bw_layers', 'bw_attn', 'bw_ffn', 'bw_norm', 'bw_head', 'bw_embed', 'bw_attn_out', 'bw_attn_score', 'bw_attn_qkv', 'bw_ffn_compute',
            'bw_layers_step', 'bw_attn_step', 'bw_ffn_step', 'bw_norm_step', 'bw_attn_out_step', 'bw_attn_score_step', 'bw_attn_qkv_step', 'bw_ffn_compute_step',
            'opt_step', 'opt_clip', 'opt_update'
        ]}
        self._memory_stats = {'grad_total_mb': 0.0, 'fwd_activation_mb': 0.0, 'grad_breakdown': {}, 'opt_per_step_mb': []}

    def _rope_backward(self, dy, positions):
        B, T, H, D = dy.shape

        try:
            cos = self.rope_cache["cos"][positions]
            sin = self.rope_cache["sin"][positions]
            if cos.shape[0] != T:
                raise IndexError("Position shape mismatch")
        except (AttributeError, IndexError, KeyError, TypeError):
            pos = np.asarray(positions, dtype=np.float64)
            if len(pos) != T:
                if len(pos) > T: pos = pos[:T]
                else: pos = np.concatenate([pos, np.arange(pos[-1]+1, pos[-1]+1 + T - len(pos))])
            inv_freq = 1.0 / (self.cfg.rope_base ** (np.arange(0, D, 2) / D))
            freqs = np.outer(pos, inv_freq)
            cos, sin = np.cos(freqs), np.sin(freqs)

        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        dy1, dy2 = dy[..., 0::2], dy[..., 1::2]
        dx = np.empty_like(dy)
        dx[..., 0::2] = dy1 * cos + dy2 * sin
        dx[..., 1::2] = -dy1 * sin + dy2 * cos
        return dx

    def sample_top_k(self, logits, k=50, temperature=1.0):
        logits = logits[:, -1, :] / (temperature + 1e-9)
        indices_to_remove = logits < np.partition(logits, -k, axis=-1)[..., -k, None]
        logits[indices_to_remove] = -float('Inf')
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in probs])

    def detokenize(self, token_ids): return [f"<token_{tid}>" for tid in token_ids]

    def _print_stats_table(self, title="TIMING BREAKDOWN (seconds)"):
        self._log("\n" + "="*50)
        self._log(f"{title:^50}")
        self._log("-" * 50)
        self._log(f"{'Phase':<25} | {'p50':<10} | {'p99':<10}")
        self._log("-" * 50)
        def pr(n, ts):
            if ts: self._log(f"{n:<25} | {np.percentile(ts, 50):<10.4f} | {np.percentile(ts, 99):<10.4f}")

        t_up = title.upper()
        if "FWD" in t_up or "TIMING BREAKDOWN" in t_up:
            pr("Embedding", self._stats.get('embedding'))
            pr("Layer Total (All)", self._stats.get('layers_step'))
            pr("  Norm", self._stats.get('norm_step'))
            pr("  Weight Load", self._stats.get('weight_load_step'))
            pr("  Attention", self._stats.get('attn_step'))
            pr("    Ld weights", self._stats.get('attn_load_step'))
            pr("    QKV Proj", self._stats.get('attn_qkv_step'))
            pr("    RoPE", self._stats.get('attn_rope_step'))
            pr("    Score/Sftmx", self._stats.get('attn_score_step'))
            pr("    Out Agg/Proj", self._stats.get('attn_out_step'))
            pr("  FeedForward", self._stats.get('ffn_step'))
            pr("    Ld weights", self._stats.get('ffn_load_step'))
            pr("    Compute", self._stats.get('ffn_compute_step'))
            pr("Output Head", self._stats.get('output_head'))
            if self._stats.get('prefill'):
                self._log("-" * 50)
                pr("Prefill Stage", self._stats['prefill'])
            if self._stats.get('decode'):
                self._log("-" * 50)
                pr("Decode Stage", self._stats['decode'])
        elif "BWD" in t_up or "BACKWARD" in t_up:
            pr("Backward Pass", self._stats.get('bw_total'))
            pr("  BW Head", self._stats.get('bw_head'))
            pr("  BW Layers (All)", self._stats.get('bw_layers_step'))
            pr("    BW Attention", self._stats.get('bw_attn_step'))
            pr("      BW Out Proj", self._stats.get('bw_attn_out_step'))
            pr("      BW Score", self._stats.get('bw_attn_score_step'))
            pr("      BW QKV", self._stats.get('bw_attn_qkv_step'))
            pr("    BW FeedForward", self._stats.get('bw_ffn_step'))
            pr("      BW SwiGLU", self._stats.get('bw_ffn_compute_step'))
            pr("    BW Norms", self._stats.get('bw_norm_step'))
            pr("  BW Embed", self._stats.get('bw_embed'))
        elif "OPT" in t_up:
            pr("Optimizer Step", self._stats.get('opt_step'))
            pr("  Grad Clipping", self._stats.get('opt_clip'))
            pr("  Weight Updates", self._stats.get('opt_update'))

        self._log("="*50 + "\n")

    def merge_lora_weights(self):
        if not self.use_lora: return
        print("[INFO] Merging LoRA adapters...")
        c = self.cfg
        scale = self.lora_alpha / self.lora_rank
        for i in range(c.n_layers):
            p = self.layer_weights[i]
            for k in ["q", "v"]:
                if p.get(f"lora_A_{k}"):
                    W_name = p[f"W_{k}"]
                    A, B = self.weights[p[f"lora_A_{k}"]], self.weights[p[f"lora_B_{k}"]]
                    self.weights[W_name] = self.weights[W_name] + scale * (A @ B).T
        self.use_lora = False
        print("[INFO] LoRA merge complete (LoRA disabled after merge).")


class EinsumOptimizer:
    def __init__(self, model: EinsumTransformer, lr=0.001, grad_clip=1.0):
        self.model, self.lr, self.grad_clip = model, lr, grad_clip

    def step(self):
        """
        Optimizer step: Apply gradient descent to update all model weights.
        HIGH-LEVEL FLOW:
        ================
        1. Embedding Update:
           - E = E - lr * dE
        
        2. For each layer (0 to N-1):
           a. Attention Weight Updates:
              - W_q = W_q - lr * dW_q^T  [Transpose to match safetensors (out, in) format]
              - W_k = W_k - lr * dW_k^T
              - W_v = W_v - lr * dW_v^T
              - W_o = W_o - lr * dW_o^T
           b. Normalization Weight Updates:
              - norm1 = norm1 - lr * d_norm1  [Element-wise, no transpose]
              - norm2 = norm2 - lr * d_norm2
           c. FFN Weight Updates:
              - W1 = W1 - lr * dW1^T  [Transpose to match safetensors format]
              - W2 = W2 - lr * dW2^T
              - W3 = W3 - lr * dW3^T
        
        GRADIENT DESCENT OPERATION:
        ===========================
        Basic SGD update rule:
          W_new = W_old - learning_rate * gradient
        
        TRANSPOSE RATIONALE:
        ====================
        - Gradients are computed in (in_features, out_features) based on einsum notation
        - Safetensors stores weights as (out_features, in_features)
        - We apply .T to gradients before subtraction to match weight tensor shape
        - Exception: Normalization weights are 1D vectors, no transpose needed
        
        EINSUM INTERPRETATION:
        ======================
        While no explicit einsum is used in weight updates, the operation is:
          W[i,j] = W[i,j] - lr * dW[i,j]  (element-wise subtraction)
        
        This could be expressed as einsum('ij,ij->ij', W, ones) - lr * dW,
        but direct array subtraction is more efficient.
        
        WEIGHT PERSISTENCE:
        ===================
        Updates are applied in-place to LazyWeightManager.cache, enabling
        iterative training without reloading weights from disk.
        """
        m, g, lr = self.model, self.model._grads, self.lr
        t0 = time.time()

        t_clip = time.time()
        # Global gradient clipping for stability
        if self.grad_clip > 0:
            total_norm = 0
            for grad in g.values():
                total_norm += np.sum(grad**2)
            total_norm = np.sqrt(total_norm)
            clip_coef = self.grad_clip / (total_norm + 1e-6)
            if clip_coef < 1:
                for k in g:
                    g[k] *= clip_coef
        m._stats['opt_clip'].append(time.time() - t_clip)

        t_update = time.time()

        # Calculate memory and update E in main thread
        step_memory_bytes = 0
        if "E" in g:
            step_memory_bytes += m.weights[m.E_name].nbytes + g["E"].nbytes
            m.weights[m.E_name] -= lr * g["E"]

        def update_layer(i):
            p = m.layer_weights[i]
            mem = 0
            for k in ["lora_A_q", "lora_B_q", "lora_A_v", "lora_B_v"]:
                w_key = f"{k}_{i}"
                if w_key in m.weights and w_key in g:
                    mem += m.weights[w_key].nbytes + g[w_key].nbytes
                    m.weights[w_key] -= lr * g[w_key]
            for k in ["W_q", "W_k", "W_v", "W_o", "W1", "W2", "W3"]:
                grad_key = f"{k}_{i}"
                if p.get(k) and grad_key in g:
                    mem += m.weights[p[k]].nbytes + g[grad_key].nbytes
                    m.weights[p[k]] -= lr * g[grad_key].T
            for k in ["norm1", "norm2"]:
                grad_key = f"{k}_{i}"
                if p.get(k) and grad_key in g:
                    mem += m.weights[p[k]].nbytes + g[grad_key].nbytes
                    m.weights[p[k]] -= lr * g[grad_key]
            return mem

        futures = [m.executor.submit(update_layer, i) for i in range(m.cfg.n_layers)]
        for f in as_completed(futures):
            step_memory_bytes += f.result()

        m._stats['opt_update'].append(time.time() - t_update)

        w_mem = m.weights.get_cache_memory_mb() if hasattr(m.weights, 'get_cache_memory_mb') else 0
        m._memory_stats['opt_per_step_mb'].append((step_memory_bytes / (1024 * 1024)) + w_mem)
        m._stats['opt_step'].append(time.time() - t0)
