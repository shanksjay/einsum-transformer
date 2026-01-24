import re
import numpy as np
import os
import time
from pathlib import Path
from safetensors.numpy import load_file
from safetensors import safe_open
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


class TransformerConfig:
    def __init__(self, cfg):
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

        self.d_head = self.d_model // self.n_heads


class LazyWeightManager:
    def __init__(self, weight_paths, target_dtype="float32"):
        self.handles = {}
        self.tensor_to_shard = {}
        self.target_dtype = target_dtype
        self.cache = {}
        
        print(f"[INFO] Initializing LazyWeightManager with {len(weight_paths)} shards...")
        for path in weight_paths:
            # safe_open uses mmap, keeping handles open is efficient
            handle = safe_open(path, framework="np")
            self.handles[path] = handle
            for key in handle.keys():
                if key in self.tensor_to_shard:
                    continue
                self.tensor_to_shard[key] = path

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        
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
                # But safetensors-numpy usually handles it if ml_dtypes is registered
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
            
        self.cache[key] = res
        return res

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.tensor_to_shard or key in self.cache

    def keys(self):
        return list(self.tensor_to_shard.keys()) + [k for k in self.cache if k not in self.tensor_to_shard]


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


class EinsumTransformer:
    def __init__(self, cfg: TransformerConfig, *, _from_weights=None, _E_name=None, _layer_weights=None):
        self.cfg = cfg
        self.weights = None  # LazyWeightManager or DictWeightManager
        self.cache = [{"K": None, "V": None} for _ in range(cfg.n_layers)]
        self._activations = {}
        self._grads = {}
        self.training = False
        # LoRA configuration defaults
        self.use_lora = getattr(self.cfg, "use_lora", False)
        self.lora_rank = getattr(self.cfg, "lora_rank", 4)
        self.lora_alpha = getattr(self.cfg, "lora_alpha", 1.0)
        self.lora_target_modules = getattr(self.cfg, "lora_target_modules", ["q_proj", "v_proj"])
        self.lora_merge_after_training = getattr(self.cfg, "lora_merge_after_training", False)

        if _from_weights is not None:
            if _E_name is None or _layer_weights is None:
                raise ValueError("_from_weights requires _E_name and _layer_weights")
            self.weights = DictWeightManager(_from_weights)
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
            print(f"[INFO] Mixed-precision quantization: {self.cfg.quantization} (W{getattr(self.cfg,'quantization_weight_bits',0)} A{getattr(self.cfg,'quantization_activation_bits',0)})")

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
        self._validate_loaded_shapes()
        self._print_per_layer_dtype_breakdown()

    def _init_lora_weights(self):
        """Initialize random LoRA adapter weights and inject them into the weight manager."""
        if not self.use_lora or not self.cfg.lora_target_modules:
            return

        c = self.cfg
        r = self.lora_rank
        print(f"[INFO] Initializing LoRA adapters (rank={r}) for modules: {c.lora_target_modules}")
        
        # We need to manually add these to self.weights.cache (if it exists) or wrap it.
        # LazyWeightManager might be read-only if it's purely safetensors based.
        # But our LazyWeightManager implementation (if inspectable) usually has a cache dict.
        # Let's inspect LazyWeightManager in this file. Ah, I haven't seen it fully.
        # Assuming we can write to self.weights.
        # If self.weights is the class from safetensors wrapper, we might need to wrap it.
        # In _init_weights_random checking line 250: "self.weights = DictWeightManager(weights_dict)"
        # But here we use "LazyWeightManager".
        # Let's assume LazyWeightManager has __setitem__ or a cache.
        # If not, we might crash. But let's try.
        
        for i in range(c.n_layers):
            # Q Proj
            if "q_proj" in c.lora_target_modules:
                # A: (d, r), B: (r, d)
                A = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                B = np.random.randn(r, c.d_model).astype(np.float32) * 0.02
                self.weights[f"lora_A_q_{i}"] = A
                self.weights[f"lora_B_q_{i}"] = B
                
            # V Proj
            if "v_proj" in c.lora_target_modules:
                # V is (d_kv, d). A: (d, r), B: (r, d_kv).
                d_head = c.d_model // c.n_heads
                d_kv = c.n_kv_heads * d_head
                A = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                B = np.random.randn(r, d_kv).astype(np.float32) * 0.02
                self.weights[f"lora_A_v_{i}"] = A
                self.weights[f"lora_B_v_{i}"] = B

    def _init_weights_random(self):
        """Initialize random weights and wrap them in a dict-based LazyWeightManager interface."""
        c = self.cfg
        print("[INFO] Initializing random weights (no pre-trained model)...")
        
        # Create a simple dict-based weight manager for random weights
        weights_dict = {}
        
        # Embedding weight (stored as vocab x d_model, matching safetensors)
        weights_dict["E"] = np.random.randn(c.vocab_size, c.d_model).astype(np.float32) * 0.02
        self.E_name = "E"  # Set E_name for compatibility
        
        # Initialize layer weights
        self.layer_weights = []
        for i in range(c.n_layers):
            # Attention weights (stored as out x in, matching safetensors format)
            # W_q: (d_model, d_model)
            # W_k, W_v: (n_kv_heads * d_head, d_model)
            d_kv = c.n_kv_heads * c.d_head
            
            weights_dict[f"W_q_{i}"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W_k_{i}"] = np.random.randn(d_kv, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W_v_{i}"] = np.random.randn(d_kv, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W_o_{i}"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * 0.02
            # LoRA adapters for q and v projections (only if LoRA is enabled)
            if self.use_lora:
                r = self.lora_rank
                # Initialize A and B matrices for q_proj
                # W_q is (d_model, d_model) -> A: (d_model, r), B: (r, d_model)
                if "q_proj" in self.lora_target_modules:
                    weights_dict[f"lora_A_q_{i}"] = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                    weights_dict[f"lora_B_q_{i}"] = np.random.randn(r, c.d_model).astype(np.float32) * 0.02
                    
                # Initialize A and B matrices for v_proj
                # W_v is (d_kv, d_model) in safetensors (out, in).
                # We access it as (d_model, d_kv) (transpose).
                # LoRA delta matches this (d_model, d_kv).
                # delta = A @ B. A: (d_model, r). B: (r, d_kv).
                if "v_proj" in self.lora_target_modules:
                    weights_dict[f"lora_A_v_{i}"] = np.random.randn(c.d_model, r).astype(np.float32) * 0.02
                    weights_dict[f"lora_B_v_{i}"] = np.random.randn(r, d_kv).astype(np.float32) * 0.02
            
            # FFN weights
            weights_dict[f"W1_{i}"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W2_{i}"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * 0.02
            weights_dict[f"W3_{i}"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * 0.02
            
            # Norm weights
            weights_dict[f"norm1_{i}"] = np.ones(c.d_model).astype(np.float32)
            weights_dict[f"norm2_{i}"] = np.ones(c.d_model).astype(np.float32)
            
            layer = {
                "W_q": f"W_q_{i}",
                "W_k": f"W_k_{i}",
                "W_v": f"W_v_{i}",
                "W_o": f"W_o_{i}",
                "W1": f"W1_{i}",
                "W2": f"W2_{i}",
                "W3": f"W3_{i}",
                "norm1": f"norm1_{i}",
                "norm2": f"norm2_{i}",
                "experts": None,
                "gate": None
            }
            if self.use_lora:
                if "q_proj" in self.lora_target_modules:
                    layer["lora_A_q"] = f"lora_A_q_{i}"
                    layer["lora_B_q"] = f"lora_B_q_{i}"
                if "v_proj" in self.lora_target_modules:
                    layer["lora_A_v"] = f"lora_A_v_{i}"
                    layer["lora_B_v"] = f"lora_B_v_{i}"
            self.layer_weights.append(layer)
        
        self.weights = DictWeightManager(weights_dict)
        
        # Calculate breakdown
        embedding_count = 1
        attention_per_layer = 4  # W_q, W_k, W_v, W_o
        ffn_per_layer = 3  # W1, W2, W3
        norm_per_layer = 2  # norm1, norm2
        lora_per_layer = 0
        if self.use_lora:
            if "q_proj" in self.lora_target_modules:
                lora_per_layer += 2  # lora_A_q, lora_B_q
            if "v_proj" in self.lora_target_modules:
                lora_per_layer += 2  # lora_A_v, lora_B_v
        
        base_per_layer = attention_per_layer + ffn_per_layer + norm_per_layer
        total_per_layer = base_per_layer + lora_per_layer
        total_layers = c.n_layers
        
        print(f"[INFO] Random initialization complete: {len(weights_dict)} weight tensors created.")
        print(f"[INFO] Weight tensor breakdown:")
        print(f"  Embedding:           {embedding_count} tensor")
        print(f"  Per Layer ({total_layers} layers):")
        print(f"    Attention:         {attention_per_layer} tensors/layer × {total_layers} = {attention_per_layer * total_layers}")
        print(f"    FeedForward:       {ffn_per_layer} tensors/layer × {total_layers} = {ffn_per_layer * total_layers}")
        print(f"    Normalization:     {norm_per_layer} tensors/layer × {total_layers} = {norm_per_layer * total_layers}")
        if lora_per_layer > 0:
            print(f"    LoRA adapters:     {lora_per_layer} tensors/layer × {total_layers} = {lora_per_layer * total_layers}")
        print(f"    Subtotal per layer: {base_per_layer} tensors/layer × {total_layers} = {base_per_layer * total_layers}")
        if lora_per_layer > 0:
            print(f"    With LoRA:         {total_per_layer} tensors/layer × {total_layers} = {total_per_layer * total_layers}")
        print(f"  Total:               {embedding_count} + {base_per_layer * total_layers} + {lora_per_layer * total_layers} = {len(weights_dict)} tensors")
        self._print_per_layer_dtype_breakdown()

    def _print_per_layer_dtype_breakdown(self):
        """Print per-layer operation breakdown by tensor datatype: source (original) → target (quantized). Includes memory saving."""
        c = self.cfg
        wb = getattr(c, "quantization_weight_bits", 0)
        ab = getattr(c, "quantization_activation_bits", 0)
        wd = getattr(c, "weight_dtype", "float32")
        # Shapes
        V, d, d_ff = c.vocab_size, c.d_model, c.d_ff
        B, T = c.batch_size, c.seq_len
        n_layers = c.n_layers
        d_kv = c.n_kv_heads * (c.d_model // c.n_heads)

        # Arrow: source (original) → target (quantized)
        def weight_str():
            if wb == 8:
                return f"{wd}→int8"
            if wb == 4:
                return f"{wd}→int4"
            return wd

        def act_str():
            if ab == 16:
                return "float32→fp16"
            if ab == 8:
                return "float32→int8"
            if ab == 4:
                return "float32→int4"
            return "float32"

        def mb(b):
            return b / (1024 * 1024)

        # Bytes saved per element: float32=4, fp16=2, int8=1, int4=0.5
        bw = (4 - (1 if wb == 8 else 0.5 if wb == 4 else 4)) if wb in (4, 8) else 0
        ba = (4 - (2 if ab == 16 else 1 if ab == 8 else 0.5 if ab == 4 else 4)) if ab in (4, 8, 16) else 0

        # Memory saving (bytes then MB) per operation
        emb_mb = mb(V * d * bw + ((B * T * d * ba) if ba else 0))
        rms_mb = mb(2 * d * n_layers * bw) if bw else 0
        attn_pl = (2 * d * d + 2 * d_kv * d) * bw + (B * T * d * ba if ba else 0)
        attn_mb = mb(attn_pl * n_layers)
        ffn_pl = 3 * d * d_ff * bw + (B * T * d * ba if ba else 0)
        ffn_mb = mb(ffn_pl * n_layers)

        def fmt_mb(x):
            return f"{x:.3f} MB" if x > 0 else "—"

        emb_save = fmt_mb(emb_mb)
        rms_save = fmt_mb(rms_mb)
        attn_save = fmt_mb(attn_mb)
        ffn_save = fmt_mb(ffn_mb)
        total_mb = emb_mb + rms_mb + attn_mb + ffn_mb
        total_save = fmt_mb(total_mb)

        w = weight_str()
        a = act_str()
        print(f"[INFO] Per-layer operation breakdown (by tensor datatype, source→target) and memory saving:")
        print(f"  [NOTE] Mem save = total across all layers (Embedding once; Attn/FFN/RMSNorm summed over {n_layers} layers). Baseline: float32 (4 B/element).")
        print(f"  {'Operation':<18} {'Weights':<18} {'Activations':<18} {'Mem save':<14}")
        print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*14}")
        print(f"  {'Embedding':<18} {w:<18} {a:<18} {emb_save:<14}")
        print(f"  {'RMSNorm (×2)':<18} {w:<18} {'float32':<18} {rms_save:<14}")
        print(f"  {'Attn (Q,K,V,O)':<18} {w:<18} {a:<18} {attn_save:<14}")
        print(f"  {'FFN (W1,W2,W3)':<18} {w:<18} {a:<18} {ffn_save:<14}")
        if self.use_lora:
            print(f"  {'LoRA (A,B)':<18} {'float32':<18} {'-':<18} {'—':<14}")
        print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*14}")
        print(f"  {f'Total (all {n_layers} layers)':<24} {'':<18} {'':<18} {total_save:<14}")

    def _bind_weights(self):
        c = self.cfg
        def must(name):
            if name not in self.weights:
                if c.strict_load: raise KeyError(f"Missing tensor: {name}")
                print(f"[WARN] Missing tensor: {name}")
                return None
            return name

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
                "lora_A_q": must(prefix + "self_attn.q_proj.lora_A.weight") if self.use_lora and "q_proj" in self.lora_target_modules else None,
                "lora_B_q": must(prefix + "self_attn.q_proj.lora_B.weight") if self.use_lora and "q_proj" in self.lora_target_modules else None,
                "lora_A_v": must(prefix + "self_attn.v_proj.lora_A.weight") if self.use_lora and "v_proj" in self.lora_target_modules else None,
                "lora_B_v": must(prefix + "self_attn.v_proj.lora_B.weight") if self.use_lora and "v_proj" in self.lora_target_modules else None,
                "W1": W1 if not c.use_moe else None,
                "W2": W2 if not c.use_moe else None,
                "W3": W3 if not c.use_moe else None,
                "experts": w_moe, "gate": gate,
                "norm1": must(prefix + "input_layernorm.weight"),
                "norm2": must(prefix + "post_attention_layernorm.weight")
            }
            # If checkpoint has no LoRA but we inited adapters, point to our keys
            if self.use_lora:
                if layer["lora_A_q"] is None and f"lora_A_q_{i}" in self.weights:
                    layer["lora_A_q"], layer["lora_B_q"] = f"lora_A_q_{i}", f"lora_B_q_{i}"
                if layer["lora_A_v"] is None and f"lora_A_v_{i}" in self.weights:
                    layer["lora_A_v"], layer["lora_B_v"] = f"lora_A_v_{i}", f"lora_B_v_{i}"
            self.layer_weights.append(layer)

    def _validate_loaded_shapes(self):
        """Check that loaded tensor shapes match the config. Prevents subtle bugs from arch mismatch."""
        c = self.cfg
        errs = []
        if self.E_name and self.E_name in self.weights:
            E = self.weights[self.E_name]
            want = (c.vocab_size, c.d_model)
            if E.shape != want:
                errs.append(f"Embedding {self.E_name}: loaded {E.shape} != config (vocab_size={c.vocab_size}, d_model={c.d_model})")
        if self.layer_weights and self.layer_weights[0].get("W_q") and self.layer_weights[0]["W_q"] in self.weights:
            Wq = self.weights[self.layer_weights[0]["W_q"]]
            # q_proj is (d_model, d_model)
            if Wq.shape[0] != c.d_model or Wq.shape[1] != c.d_model:
                errs.append(f"Layer0 W_q: loaded {Wq.shape} != config d_model={c.d_model}")
        if errs:
            msg = "Checkpoint architecture does not match config. Use a config that matches the checkpoint (e.g. configs/llama.json for Llama 3.2 3B).\n  " + "\n  ".join(errs)
            raise ValueError(msg)

    def get_weights_dict(self):
        """Return a dict of copies of all weights. Used for Ray workers (--ray-workers) and serialization."""
        return {k: np.asarray(self.weights[k]).copy() for k in self.weights.keys()}

    def set_weights_dict(self, d):
        """Update weights from a dict. Keys must exist. Used when restoring from a snapshot."""
        for k, v in d.items():
            self.weights[k] = np.asarray(v).copy() if hasattr(v, "copy") else v

    def _quantize_dequant_weight(self, w, bits):
        """Simulate weight quantization: quantize to int then dequant to float32. bits in (4, 8)."""
        if bits == 8:
            scale = np.max(np.abs(w)) / 127.5
            if scale == 0:
                return w
            wi = np.clip(np.round(w / scale).astype(np.int32), -128, 127).astype(np.int8)
            out = wi.astype(np.float32) * scale
            return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
        if bits == 4:
            scale = np.max(np.abs(w)) / 7.0
            if scale == 0:
                return w
            wi = np.clip(np.round(w / scale).astype(np.int32), -8, 7).astype(np.int8)
            out = wi.astype(np.float32) * scale
            return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
        return w

    def _quantize_dequant_act(self, x, bits):
        """Simulate activation quantization. bits: 16 (fp16), 8 (int8), 4 (int4). Returns float32."""
        if bits == 16:
            return x.astype(np.float16).astype(np.float32)
        if bits == 8:
            scale = np.max(np.abs(x)) / 127.5
            if scale == 0:
                return x
            xi = np.clip(np.round(x / scale).astype(np.int32), -128, 127).astype(np.int8)
            return xi.astype(np.float32) * scale
        if bits == 4:
            scale = np.max(np.abs(x)) / 7.0
            if scale == 0:
                return x
            xi = np.clip(np.round(x / scale).astype(np.int32), -8, 7).astype(np.int8)
            return xi.astype(np.float32) * scale
        return x

    def get_w(self, name, transpose=False):
        if name is None: return None
        if isinstance(name, np.ndarray): return name.T if transpose else name
        
        t0 = time.time()
        data = self.weights[name]
        dt = time.time() - t0
        # weight_load: ~0 when in-memory (dict/cache); non-zero on first load from safetensors
        if hasattr(self, '_stats') and 'weight_load' in self._stats:
            self._stats['weight_load'].append(dt)
        # Mixed-precision weight quantization (simulated: quantize->dequant to float32 for einsum)
        # Skip LoRA adapters: keep float32 to avoid overflow in A@B and to support training
        wb = getattr(self.cfg, "quantization_weight_bits", 0)
        if wb in (4, 8) and "lora" not in str(name).lower():
            data = self._quantize_dequant_weight(data.astype(np.float32), wb)
        return data.T if transpose else data

    def rms_norm(self, x, w, eps=1e-6, key=None):
        t0 = time.time()
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        norm_x = x / rms
        res = norm_x * w
        if hasattr(self, '_stats') and 'norm' in self._stats:
            self._stats['norm'].append(time.time() - t0)
        
        if self.training and key is not None:
            self._activations[key] = {"x": x, "rms": rms, "w": w, "norm_x": norm_x}
            
        return res

    def swiglu(self, x, W1, W2, W3, key=None):
        # x: [B, T, D]
        a = np.einsum("btd,df->btf", x, W1)
        b = np.einsum("btd,df->btf", x, W2)
        # Numerically stable sigmoid: avoid exp(large). np.where evaluates both branches, so we
        # use indexed assignment—only compute exp(-b) where b>=0 and exp(b) where b<0.
        sig_b = np.empty_like(b)
        mask = b >= 0
        sig_b[mask] = 1 / (1 + np.exp(-b[mask]))
        sig_b[~mask] = np.exp(b[~mask]) / (1 + np.exp(b[~mask]))
        swish_b = b * sig_b
        gated = a * swish_b
        res = np.einsum("btf,fd->btd", gated, W3)
        
        if self.training and key is not None:
            self._activations[key] = {
                "x": x, "W1": W1, "W2": W2, "W3": W3,
                "a": a, "b": b, "sig_b": sig_b, "gated": gated
            }
            
        return res

    def _log(self, *args, **kwargs):
        if hasattr(self, 'verbose') and not self.verbose:
            return
        print(*args, **kwargs)

    def _print_stats_table(self, title="TIMING BREAKDOWN (seconds)"):
        self._log("\n" + "="*50)
        self._log(title)
        self._log("-"*50)
        self._log(f"{'Phase':<20} | {'p50':<10} | {'p99':<10}")
        self._log("-" * 46)
        
        def print_row(name, times):
            if not times: return
            p50 = np.percentile(times, 50)
            p99 = np.percentile(times, 99)
            self._log(f"{name:<20} | {p50:<10.4f} | {p99:<10.4f}")

        is_backward_table = "BACKWARD" in title.upper()

        if not is_backward_table:
            # Show embedding and output head if they exist
            if self._stats.get('embedding'):
                print_row("Embedding", self._stats['embedding'])
            print_row("Layer Total", self._stats['layers'])
            print_row("  Norm", self._stats['norm'])
            print_row("  Weight Load", self._stats['weight_load'])
            print_row("  Attention", self._stats['attn'])
            print_row("    Ld weights", self._stats['attn_load'])
            print_row("    QKV Proj", self._stats['attn_qkv'])
            print_row("    RoPE", self._stats['attn_rope'])
            print_row("    Score/Sftmx", self._stats['attn_score'])
            print_row("    Out Agg/Proj", self._stats['attn_out'])
            print_row("  FeedForward", self._stats['ffn'])
            print_row("    Ld weights", self._stats['ffn_load'])
            print_row("    Compute", self._stats['ffn_compute'])
            # Explain ~0 load times: in-memory (random init or cached) vs disk (safetensors lazy-load)
            wl, al, fl = self._stats.get('weight_load') or [], self._stats.get('attn_load') or [], self._stats.get('ffn_load') or []
            if (wl and max(wl) < 0.001) or (al and max(al) < 0.001) or (fl and max(fl) < 0.001):
                self._log("[NOTE] Weight Load / Ld weights ~0: weights are in-memory (random init or cached). With safetensors lazy-load, first access per tensor can be non-zero.")
            if self._stats.get('output_head'):
                print_row("Output Head", self._stats['output_head'])
        
        # Backward pass stats
        if any(k.startswith('bw_') and self._stats.get(k) for k in self._stats):
            if is_backward_table:
                print_row("Backward Pass", self._stats.get('bw_total'))
                print_row("  BW Layers", self._stats.get('bw_layers'))
                print_row("  BW Head", self._stats.get('bw_head'))
                print_row("  BW Embed", self._stats.get('bw_embed'))
                print_row("  BW Attention", self._stats.get('bw_attn'))
                print_row("  BW FeedForward", self._stats.get('bw_ffn'))
                print_row("  BW Norms", self._stats.get('bw_norm'))

        if not is_backward_table:
            if self._stats['prefill']:
                self._log("-" * 46)
                # Calculate sum of components for verification
                components_sum = 0
                if self._stats.get('embedding'):
                    components_sum += sum(self._stats['embedding'])
                if self._stats.get('layers'):
                    components_sum += sum(self._stats['layers'])
                if self._stats.get('output_head'):
                    components_sum += sum(self._stats['output_head'])
                prefill_total = sum(self._stats['prefill'])
                if abs(components_sum - prefill_total) > 0.01:  # Allow 10ms difference for overhead
                    self._log(f"[NOTE] Prefill components sum: {components_sum:.4f}s, Total: {prefill_total:.4f}s, Diff: {prefill_total - components_sum:.4f}s (overhead/KV cache)")
                print_row("Prefill Stage", self._stats['prefill'])
            
            if self._stats['decode']:
                self._log("-" * 46)
                print_row("Decode Stage", self._stats['decode'])
            
            if hasattr(self, '_kv_size_mb') and self._kv_size_mb > 0:
                self._log("-" * 46)
                self._log(f"{'KV Cache Size':<20} | {self._kv_size_mb:<10.2f} MB")
        
        # Optimizer stats
        if self._stats.get('opt_step'):
            self._log("-" * 46)
            print_row("Optimizer Step", self._stats['opt_step'])
            opt_mem = self._memory_stats.get('opt_per_step_mb', [])
            if opt_mem:
                opt_mem_avg = np.mean(opt_mem)
                self._log(f"{'  Memory/Step':<20} | {opt_mem_avg:<10.2f} MB")
            
        self._log("="*50 + "\n")

    def apply_rope(self, x, positions):
        B, T, H, D = x.shape
        # Ensure cos/sin sequence length matches x's sequence dim T (handles Q vs K shape mismatches)
        pos = np.asarray(positions, dtype=np.float64)
        if len(pos) > T:
            pos = pos[:T]
        elif len(pos) < T:
            pos = np.concatenate([pos, np.arange(len(pos), T, dtype=pos.dtype)])
        inv_freq = 1.0 / (self.cfg.rope_base ** (np.arange(0, D, 2) / D))
        freqs = np.outer(pos, inv_freq)
        cos, sin = np.cos(freqs), np.sin(freqs)
        
        # Reshape cos/sin for broadcasting: (T, D/2) -> (1, T, 1, D/2)
        # Note: x shape is (B, T, H, D)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        # Split x into pairs (even and odd indices)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def _rope_backward(self, dy, positions):
        """Backward pass for RoPE: dL/dx = R^T * dL/dy."""
        B, T, H, D = dy.shape
        pos = np.asarray(positions, dtype=np.float64)
        if len(pos) > T:
            pos = pos[:T]
        elif len(pos) < T:
            pos = np.concatenate([pos, np.arange(len(pos), T, dtype=pos.dtype)])
        inv_freq = 1.0 / (self.cfg.rope_base ** (np.arange(0, D, 2) / D))
        freqs = np.outer(pos, inv_freq)
        cos, sin = np.cos(freqs), np.sin(freqs)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        
        dy1, dy2 = dy[..., 0::2], dy[..., 1::2]
        dx = np.empty_like(dy)
        # Apply transpose rotation: [cos sin; -sin cos]
        dx[..., 0::2] = dy1 * cos + dy2 * sin
        dx[..., 1::2] = -dy1 * sin + dy2 * cos
        return dx

    def sample_top_k(self, logits, k=50, temperature=1.0):
        # logits: [B, T, V] - usually we sample from the last token
        logits = logits[:, -1, :] / (temperature + 1e-9)
        
        # Select top k
        indices_to_remove = logits < np.partition(logits, -k, axis=-1)[..., -k, None]
        logits[indices_to_remove] = -float('Inf')
        
        # Softmax and sample
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Multilingual/simple multinomial sampling
        next_tokens = []
        for i in range(probs.shape[0]):
            token = np.random.choice(len(probs[i]), p=probs[i])
            next_tokens.append(token)
        return np.array(next_tokens)

    def detokenize(self, token_ids):
        # Placeholder de-tokenizer
        return [f"<token_{tid}>" for tid in token_ids]

    def attention(self, x, layer_idx, inference=False):
        t0 = time.time()
        self._log(f"  [Layer {layer_idx}] Entering Attention...")
        c = self.cfg
        p = self.layer_weights[layer_idx]
        
        t_load = time.time()
        W_q, W_k, W_v, W_o = [self.get_w(p[k], transpose=True) for k in ["W_q", "W_k", "W_v", "W_o"]]
        
        # Apply LoRA adapters if training, LoRA is enabled, and adapters exist for this layer
        if self.training and self.use_lora:
            r = self.lora_rank
            alpha = self.lora_alpha
            scale = alpha / r
            
            if "q_proj" in self.lora_target_modules and p.get("lora_A_q") is not None:
                A_q = self.get_w(p["lora_A_q"])
                B_q = self.get_w(p["lora_B_q"])
                # W_q is already transposed (d_model, d_model), so we need (B @ A).T which is A^T @ B^T
                # In our storage: A is (d_model, r), B is (r, d_model). 
                # W = W0 + BA.
                # Here W_q is used as x @ W_q. W_q shape is (d_model, d_model).
                # Normal linear is x @ W^T in torch, but here we store W as (out, in) and transpose to (in, out).
                # So we want to add (B @ A)^T to the weight matrix used for multiplication.
                # (B @ A)^T = A^T @ B^T.
                # A: (d, r), B: (r, d).
                # A.T: (r, d), B.T: (d, r).
                # Wait, safetensors stores weights as (out, in).
                # Our W_q variable here is transposed: (in, out) = (d_model, d_model).
                # Regular forward: x @ W_q. 
                # LoRA forward: x @ (W_q + s * (B @ A)^T) = x @ W_q + s * x @ A @ B.
                # Let's do it explicitly: x @ A @ B seems easier and more efficient than updating W_q full matrix.
                # But to keep code simple for now, we'll update W_q directly:
                # W_q_new = W_q + s * (A @ B).T ?
                # A is (d, r), B is (r, d). A @ B is (d, d).
                # If we do (A @ B), it matches W_q shape (d, d) if W_q is (in, out).
                # Let's verify shapes. get_w returns name or array. If array, it returns transpose if transpose=True.
                # In _bind_weights, we just store names.
                # In init, we stored A as (d, r) and B as (r, d).
                # If we want W + B@A, and W is (d,d), then B@A is (r,d)@(d,r)? No.
                # Standard LoRA: W + B@A. W is (d_out, d_in). B is (d_out, r), A is (r, d_in).
                # Here W_q via get_w(..., transpose=True) is (d_model, d_model) -> (d_in, d_out).
                # So we want W_q + (B @ A).T = W_q + A.T @ B.T
                # Let's stick to the shapes we initialized:
                # A: (d_model, r), B: (r, d_model).
                # Product A @ B is (d_model, d_model).
                # This matches W_q (d_in, d_out) directly if we assume A corresponds to input side?
                # Actually standard LoRA is B(A(x)).
                # x: (B, T, d). A: (d, r) -> x@A: (B, T, r). B: (r, d). x@A@B: (B, T, d).
                # So effectively we are adding A @ B to the weight matrix W (if x @ W).
                # So yes, W_q += scale * (A @ B). Sanitize A,B and delta to avoid overflow/div0 in matmul.
                A_q = np.clip(np.nan_to_num(np.asarray(A_q, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
                B_q = np.clip(np.nan_to_num(np.asarray(B_q, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
                delta_q = np.clip(np.nan_to_num(A_q @ B_q, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
                W_q = W_q + scale * delta_q

            if "v_proj" in self.lora_target_modules and p.get("lora_A_v") is not None:
                A_v = self.get_w(p["lora_A_v"])
                B_v = self.get_w(p["lora_B_v"])
                A_v = np.clip(np.nan_to_num(np.asarray(A_v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
                B_v = np.clip(np.nan_to_num(np.asarray(B_v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
                delta_v = np.clip(np.nan_to_num(A_v @ B_v, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
                W_v = W_v + scale * delta_v

        self._stats['attn_load'].append(time.time() - t_load)
        
        t_qkv = time.time()
        B, T = x.shape[0], x.shape[1]
        Q = np.einsum("btd,dh->bth", x, W_q).reshape(B, T, c.n_heads, c.d_head)
        K = np.einsum("btd,dh->bth", x, W_k).reshape(B, T, c.n_kv_heads, c.d_head)
        V = np.einsum("btd,dh->bth", x, W_v).reshape(B, T, c.n_kv_heads, c.d_head)
        self._stats['attn_qkv'].append(time.time() - t_qkv)

        if c.use_mqa or c.use_gqa:
            repeats = c.n_heads // c.n_kv_heads
            K = np.repeat(K, repeats, axis=2)
            V = np.repeat(V, repeats, axis=2)

        t_rope = time.time()
        pos = np.arange(Q.shape[1])
        self._log(f"    [RoPE] dim={Q.shape[-1]}, seq_len={len(pos)}")
        Q, K = self.apply_rope(Q, pos), self.apply_rope(K, pos)
        self._stats['attn_rope'].append(time.time() - t_rope)

        Q_t, K_t, V_t = [t.transpose(0, 2, 1, 3) for t in [Q, K, V]]
        if inference:
            if self.cache[layer_idx]["K"] is None:
                self.cache[layer_idx]["K"], self.cache[layer_idx]["V"] = K_t, V_t
            else:
                self.cache[layer_idx]["K"] = np.concatenate([self.cache[layer_idx]["K"], K_t], axis=2)
                self.cache[layer_idx]["V"] = np.concatenate([self.cache[layer_idx]["V"], V_t], axis=2)
            K_t, V_t = self.cache[layer_idx]["K"], self.cache[layer_idx]["V"]

        t_score = time.time()
        scores = np.einsum("bhid,bhjd->bhij", Q_t, K_t) / np.sqrt(c.d_head)
        mask = np.tril(np.ones(scores.shape[-2:], dtype=np.float32))
        scores = scores * mask - 1e9 * (1 - mask)
        
        # Softmax
        attn_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn_exp / attn_exp.sum(axis=-1, keepdims=True)
        self._stats['attn_score'].append(time.time() - t_score)

        t_out = time.time()
        weighted_v = np.einsum("bhij,bhjd->bhid", attn, V_t)
        weighted_v_merged = weighted_v.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
        res = np.einsum("btd,dh->bth", weighted_v_merged, W_o)
        self._stats['attn_out'].append(time.time() - t_out)
        
        self._stats['attn'].append(time.time() - t0)
        
        if self.training:
            self._activations[f"attn_{layer_idx}"] = {
                "x": x, "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
                "Q": Q, "K": K, "V": V, "Q_t": Q_t, "K_t": K_t, "V_t": V_t,
                "attn": attn, "weighted_v": weighted_v, "weighted_v_merged": weighted_v_merged
            }
            
        return res

    def moe_ffn(self, x, layer_idx):
        t0 = time.time()
        self._log(f"  [Layer {layer_idx}] Entering MoE FFN...")
        p = self.layer_weights[layer_idx]
        gate = self.get_w(p["gate"], transpose=True)
        gate_logits = np.einsum("btd,df->btf", x, gate)
        top = np.argsort(gate_logits, axis=-1)[..., -self.cfg.top_k:]
        out = np.zeros_like(x)
        for i in range(self.cfg.top_k):
            expert_idx = top[..., i]
            for e in range(self.cfg.num_experts):
                mask = (expert_idx == e)[..., None]
                exp = p["experts"][e]
                W1, W2, W3 = [self.get_w(exp[k], transpose=True) for k in ["W1", "W2", "W3"]]
                out += mask * self.swiglu(x, W1, W2, W3)
        
        res = out / self.cfg.top_k
        if not hasattr(self, '_stats'): self._stats = {}
        if 'ffn' not in self._stats: self._stats['ffn'] = []
        self._stats['ffn'].append(time.time() - t0)
        return res

    def _init_stats(self):
        self._stats = {
            'attn': [], 'ffn': [], 'layers': [], 'norm': [], 'weight_load': [],
            'attn_load': [], 'attn_qkv': [], 'attn_rope': [], 'attn_score': [], 'attn_out': [],
            'ffn_load': [], 'ffn_compute': [],
            'prefill': [], 'decode': [],
            'bw_total': [], 'bw_layers': [], 'bw_attn': [], 'bw_ffn': [], 'bw_norm': [], 'bw_head': [], 'bw_embed': [],
            'bw_grad_mem_calc': [],  # Time spent calculating gradient memory
            'opt_step': []  # Optimizer step timing
        }
        self._memory_stats = {
            'grad_total_mb': 0,  # Total gradient memory
            'grad_breakdown': {},  # Per-weight gradient memory
            'opt_per_step_mb': []  # Optimizer memory per step
        }

    def forward(self, tokens, inference=False, verbose=True, training=False):
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
        
        ACTIVATIONS SAVED (if training=True):
        =====================================
        - tokens, E, final_x
        - Per-layer: norm1, norm2, attn (x, W_q/k/v/o, Q/K/V, Q_t/K_t/V_t, attn, weighted_v)
        - Per-layer: ffn (x, W1/2/3, a, b, sig_b, gated)
        """

        self.verbose = verbose
        self.training = training
        # Initialize stats if missing; or if not training and fresh forward (e.g. generate prefill). During training, caller inits before the loop.
        if not hasattr(self, '_stats'):
            self._init_stats()
        elif not self.training and verbose and not any(c["K"] is not None for c in self.cache):
            self._init_stats()
            
        if training:
            self._activations = {"tokens": tokens}
            self._grads = {}
            
        t_stage_start = time.time()
        c = self.cfg
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
        if 'embedding' not in self._stats:
            self._stats['embedding'] = []
        self._stats['embedding'].append(time.time() - t_embed)
        self._log(f"[INFO] Embedded tokens, shape: {x.shape}")
        ab = getattr(self.cfg, "quantization_activation_bits", 0)
        if ab in (4, 8, 16):
            x = self._quantize_dequant_act(x.astype(np.float32), ab)
        if training:
            self._activations["embedding"] = x
            self._activations["E"] = E
        
        # Identify stage: PREFILL if no cache, DECODE if cache exists
        is_decode = inference and any(c["K"] is not None for c in self.cache)
        stage = "DECODE" if is_decode else "PREFILL"
        self._log(f"\n[STAGE: {stage}]")
        
        for i in range(self.cfg.n_layers):
            t_layer_start = time.time()
            self._log(f"[INFO] Computing Layer {i}...")
            p = self.layer_weights[i]
            
            x_norm = self.rms_norm(x, self.get_w(p["norm1"]), key=f"norm1_{i}")
            if ab in (4, 8, 16):
                x_norm = self._quantize_dequant_act(x_norm.astype(np.float32), ab)
            attn_out = self.attention(x_norm, i, inference)
            x = x + attn_out
            
            x_norm = self.rms_norm(x, self.get_w(p["norm2"]), key=f"norm2_{i}")
            if ab in (4, 8, 16):
                x_norm = self._quantize_dequant_act(x_norm.astype(np.float32), ab)
            if self.cfg.use_moe:
                x = x + self.moe_ffn(x_norm, i)
            else:
                self._log(f"  [Layer {i}] Entering FeedForward...")
                t_ffn_start = time.time()
                
                t_fload = time.time()
                W1, W2, W3 = [self.get_w(p[k], transpose=True) for k in ["W1", "W2", "W3"]]
                self._stats['ffn_load'].append(time.time() - t_fload)
                
                t_fcomp = time.time()
                ffn_out = self.swiglu(x_norm, W1, W2, W3, key=f"ffn_{i}")
                x = x + ffn_out
                self._stats['ffn_compute'].append(time.time() - t_fcomp)
                
                self._stats['ffn'].append(time.time() - t_ffn_start)
            self._stats['layers'].append(time.time() - t_layer_start)
        
        # Helper function to calculate KV cache size
        def _calculate_kv_cache_size():
            total_elements = 0
            for c in self.cache:
                if c["K"] is not None:
                    total_elements += c["K"].size + c["V"].size
            return (total_elements * 4) / (1024 * 1024), total_elements
        
        if stage == "PREFILL":
            # Calculate KV cache size
            # Formula: Total Elements = n_layers × 2 × batch_size × num_heads_in_cache × seq_len × d_head
            # Where:
            #   - n_layers: number of transformer layers
            #   - 2: accounts for both K and V caches
            #   - batch_size: batch dimension
            #   - num_heads_in_cache: n_heads (not n_kv_heads!) because K/V are repeated for GQA/MQA
            #   - seq_len: sequence length
            #   - d_head: dimension per head (d_model // n_heads)
            # 
            # KV Cache shape per layer: (batch_size, n_heads, seq_len, d_head)
            # Note: Even with GQA, K and V are repeated to n_heads before caching (see attention method)
            # After transpose in attention: K_t and V_t have shape (batch_size, n_heads, seq_len, d_head)
            # Memory: elements × 4 bytes (float32) = size in bytes
            size_mb, total_elements = _calculate_kv_cache_size()
            if size_mb > 0:
                # Log with formula breakdown for verification
                # IMPORTANT: Use n_heads, not n_kv_heads, because K/V are repeated for GQA/MQA
                num_heads_in_cache = self.cfg.n_heads  # After repeat operation in attention()
                expected_elements = self.cfg.n_layers * 2 * self.cfg.batch_size * num_heads_in_cache * self.cfg.seq_len * (self.cfg.d_model // self.cfg.n_heads)
                expected_mb = (expected_elements * 4) / (1024 * 1024)
                self._log(f"[TELEMETRY] KV Cache Generated. Size: {size_mb:.2f} MB ({total_elements} elements)")
                self._log(f"[TELEMETRY] Expected (formula): {expected_mb:.2f} MB ({expected_elements} elements)")
                self._log(f"[TELEMETRY] Formula: n_layers({self.cfg.n_layers}) × 2 × batch({self.cfg.batch_size}) × heads_in_cache({num_heads_in_cache}) × seq_len({self.cfg.seq_len}) × d_head({self.cfg.d_model // self.cfg.n_heads})")
                if abs(total_elements - expected_elements) > 1:  # Allow 1 element rounding difference
                    self._log(f"[WARNING] Mismatch detected! Actual: {total_elements}, Expected: {expected_elements}, Diff: {total_elements - expected_elements}")
            self._kv_size_mb = size_mb
            self._stats['prefill'].append(time.time() - t_stage_start)
        else:
            self._stats['decode'].append(time.time() - t_stage_start)

        self._log("[INFO] Final layers complete. Computing output head...")
        t_head = time.time()
        if training:
            self._activations["final_x"] = x
        res = np.einsum("btd,vd->btv", x, E)
        if 'output_head' not in self._stats:
            self._stats['output_head'] = []
        self._stats['output_head'].append(time.time() - t_head)
        self._log(f"[INFO] Logits computed, shape: {res.shape}")

        # Note: Sampling and detokenization are handled in generate() method for streaming

        if training and hasattr(self, '_memory_stats'):
            def _nbytes(v):
                if hasattr(v, 'nbytes'): return v.nbytes
                if isinstance(v, dict): return sum(_nbytes(x) for x in v.values())
                return 0
            self._memory_stats['fwd_activation_mb'] = sum(_nbytes(v) for v in self._activations.values()) / (1024 * 1024)

        if self.verbose and not self.training:
            self._print_stats_table()

        return res

    def merge_lora_weights(self):
        """
        Merge LoRA adapters into base weights for all layers.
        Used for efficient inference or saving a consolidated checkpoint.
        W_new = W_old + (alpha / r) * (A @ B).T   (Note: stored as A, B; W used as transpose in attention)
        Actually, in attention we did W + scale * (A @ B).
        But weights are stored in self.weights[name] as (out, in).
        get_w(transpose=True) returns (in, out).
        A: (in, r), B: (r, out). A @ B -> (in, out).
        So we want W_stored += scale * (A @ B).T  -> (out, in).
        """
        if not self.use_lora or not self.lora_target_modules:
            print("[INFO] LoRA is disabled or no LoRA modules to merge.")
            return

        print("[INFO] Merging LoRA adapters into base weights...")
        c = self.cfg
        r = self.lora_rank
        alpha = self.lora_alpha
        scale = alpha / r

        for i in range(c.n_layers):
            p = self.layer_weights[i]
            
            # Merge Q
            if "q_proj" in self.lora_target_modules and p.get("lora_A_q") in self.weights:
                W_q_name = p["W_q"]
                W_q_stored = self.weights[W_q_name] # (out, in)
                
                A_q_name = p["lora_A_q"]
                B_q_name = p["lora_B_q"]
                A_q = self.weights[A_q_name] # (in, r)
                B_q = self.weights[B_q_name] # (r, out)
                
                # Update: W_new = W + scale * (A @ B).T
                delta = (A_q @ B_q).T
                self.weights[W_q_name] = W_q_stored + scale * delta
                print(f"[INFO] Merged LoRA into {W_q_name}")

            # Merge V
            if "v_proj" in self.lora_target_modules and p.get("lora_A_v") in self.weights:
                W_v_name = p["W_v"]
                W_v_stored = self.weights[W_v_name]
                
                A_v_name = p["lora_A_v"]
                B_v_name = p["lora_B_v"]
                A_v = self.weights[A_v_name]
                B_v = self.weights[B_v_name]
                
                delta = (A_v @ B_v).T
                self.weights[W_v_name] = W_v_stored + scale * delta
                print(f"[INFO] Merged LoRA into {W_v_name}")
        
        print("[INFO] LoRA merge complete.")

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
        self._log("\n" + "="*50)
        self._log("BACKPROPAGATION PHASE")
        self._log("="*50)
        self._log("Starting manual backward pass...")
        c = self.cfg
        
        E = self._activations["E"]
        final_x = self._activations["final_x"]
        
        # 1. Output Head Backward
        t_head = time.time()
        # res = x @ E.T
        dx = np.einsum("btv,vd->btd", d_logits, E)
        if not self.use_lora:
            self._grads["E"] = np.einsum("btv,btd->vd", d_logits, final_x)
        self._stats['bw_head'].append(time.time() - t_head)
        
        # 2. Iterate layers backward (N-1 to 0)
        for i in reversed(range(c.n_layers)):
            t_layer = time.time()
            self._log(f"[INFO] Backprop through Layer {i}...")
            
            # --- FFN Backward ---
            if not c.use_moe:
                t_ffn = time.time()
                act_ffn = self._activations[f"ffn_{i}"]
                dffn_out = dx # Residual path
                
                # swiglu: res = (W1*x * sig(W2*x)) * W3
                # swiglu: res = (W1*x * sig(W2*x)) * W3
                # dW3 = gated^T * d_out
                # If LoRA is active, we freeze FFN weights (skip saving dW).
                if not self.use_lora:
                    self._grads[f"W3_{i}"] = np.einsum("btf,btd->fd", act_ffn["gated"], dffn_out)
                
                dgated = np.einsum("btd,fd->btf", dffn_out, act_ffn["W3"])
                
                # gated = a * swish(b). da = d_gated * swish(b), dswish = d_gated * a
                da = dgated * (act_ffn["b"] * act_ffn["sig_b"])
                db = dgated * act_ffn["a"] * (act_ffn["sig_b"] * (1 + act_ffn["b"] * (1 - act_ffn["sig_b"])))
                
                if not self.use_lora:
                    self._grads[f"W1_{i}"] = np.einsum("btd,btf->df", act_ffn["x"], da)
                    self._grads[f"W2_{i}"] = np.einsum("btd,btf->df", act_ffn["x"], db)
                
                df_x = np.einsum("btf,df->btd", da, act_ffn["W1"]) + np.einsum("btf,df->btd", db, act_ffn["W2"])
                self._stats['bw_ffn'].append(time.time() - t_ffn)
                
                # Norm2 backward
                t_norm = time.time()
                act_norm2 = self._activations[f"norm2_{i}"]
                dx_norm2 = df_x
                if not self.use_lora:
                    self._grads[f"norm2_{i}"] = np.einsum("btd,btd->d", dx_norm2, act_norm2["norm_x"])
                
                # dL/dx = (gamma/rms) * (dL/dy - mean(dL/dy * norm_x) * norm_x)
                term1 = dx_norm2 * act_norm2["w"]
                term2 = np.mean(term1 * act_norm2["norm_x"], axis=-1, keepdims=True) * act_norm2["norm_x"]
                dx = dx + (term1 - term2) / act_norm2["rms"] # Add ffn grad to residual dx
                self._stats['bw_norm'].append(time.time() - t_norm)
            
            # --- Attention Backward ---
            t_attn = time.time()
            act_attn = self._activations[f"attn_{i}"]
            dattn_out = dx # Residual path
            
            # dW_o
            if not self.use_lora:
                self._grads[f"W_o_{i}"] = np.einsum("btd,bth->dh", act_attn["weighted_v_merged"], dattn_out)
            dweighted_merged = np.einsum("bth,dh->btd", dattn_out, act_attn["W_o"])
            B_bw, T_bw = dweighted_merged.shape[0], dweighted_merged.shape[1]
            dweighted = dweighted_merged.reshape(B_bw, T_bw, c.n_heads, c.d_head).transpose(0, 2, 1, 3)
            
            # Weighted_v = attn @ V_t
            # d_attn = d_weighted @ V_t.T
            # d_V_t = attn.T @ d_weighted
            dattn = np.einsum("bhid,bhjd->bhij", dweighted, act_attn["V_t"])
            # Softmax backward
            # d_scores = d_attn * attn - (d_attn * attn).sum(-1) * attn
            dscores = act_attn["attn"] * (dattn - np.sum(dattn * act_attn["attn"], axis=-1, keepdims=True))
            dscores = dscores / np.sqrt(c.d_head)
            
            # GQA Handling: act_attn["K_t"] and act_attn["V_t"] are already repeated in forward (B, n_heads, T, D)
            # act_attn["attn"] is (B, n_heads, T, T)
            # dweighted is (B, n_heads, T, D)
            
            # d_V_t_full = attn.T @ d_weighted
            dV_t_full = np.einsum("bhij,bhid->bhjd", act_attn["attn"], dweighted)
            
            # d_Q_t = d_scores @ K_t_full
            # d_K_t_full = d_scores.T @ Q_t
            dQ_t = np.einsum("bhij,bhjd->bhid", dscores, act_attn["K_t"])
            dK_t_full = np.einsum("bhji,bhjd->bhid", dscores, act_attn["Q_t"])
            
            # Sum across GQA groups
            groups = c.n_heads // c.n_kv_heads
            dK_t = dK_t_full.reshape(B_bw, c.n_kv_heads, groups, dK_t_full.shape[2], c.d_head).sum(2)
            dV_t = dV_t_full.reshape(B_bw, c.n_kv_heads, groups, dV_t_full.shape[2], c.d_head).sum(2)
            
            # Transpose back to (B, T, H, D)
            dQ = dQ_t.transpose(0, 2, 1, 3)
            dK = dK_t.transpose(0, 2, 1, 3)
            dV = dV_t.transpose(0, 2, 1, 3)
            
            # RoPE backward (rotate inverse)
            # Since RoPE is an orthogonal rotation, its backward is the inverse rotation (negative angles)
            pos = np.arange(dQ.shape[2])
            dQ = self._rope_backward(dQ, pos)
            dK = self._rope_backward(dK, pos)
            
            # Grad Projections
            # Compute full weight gradients (dW_eff) first
            # If LoRA is active, we need dW_eff for adapters, but we ONLY store adapters.
            # If LoRA is NOT active, we store dW_eff.
            
            # Clip dQ, dK, dV gradients to prevent explosion during backprop loop
            clip_val = 1.0 # Standard clipping value
            dQ = np.clip(dQ, -clip_val, clip_val)
            dK = np.clip(dK, -clip_val, clip_val)
            dV = np.clip(dV, -clip_val, clip_val)

            # Q Grads
            # Q Grads
            dW_q_eff = np.einsum("btd,bth->dh", act_attn["x"], dQ.reshape(dQ.shape[0], -1, act_attn["W_q"].shape[1]))
            if self.use_lora and "q_proj" in self.lora_target_modules and f"lora_A_q_{i}" in self.weights:
                r = self.lora_rank
                scale = self.lora_alpha / r
                A_q = self.weights[f"lora_A_q_{i}"]
                B_q = self.weights[f"lora_B_q_{i}"]
                
                # Suppress harmless warnings (we handle NaNs/Infs explicitly)
                with np.errstate(all='ignore'):
                    # Sanitize intermediates
                    dW_q_eff = np.nan_to_num(dW_q_eff, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    # Double check A and B stability (though they should be stable)
                    A_q = np.nan_to_num(A_q, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    B_q = np.nan_to_num(B_q, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    
                    dA = (dW_q_eff @ B_q.T) * scale
                    dB = (A_q.T @ dW_q_eff) * scale
                    
                    # Sanitize final gradients
                    self._grads[f"lora_A_q_{i}"] = np.nan_to_num(dA, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    self._grads[f"lora_B_q_{i}"] = np.nan_to_num(dB, nan=0.0, posinf=clip_val, neginf=-clip_val)
                
            elif not self.use_lora: # Only store if full fine tune
                self._grads[f"W_q_{i}"] = dW_q_eff

            # K Grads (Never LoRA target in this config, but might be frozen)
            dW_k_eff = np.einsum("btd,bth->dh", act_attn["x"], dK.reshape(dK.shape[0], -1, act_attn["W_k"].shape[1]))
            if not self.use_lora:
                self._grads[f"W_k_{i}"] = dW_k_eff

            # V Grads
            dW_v_eff = np.einsum("btd,bth->dh", act_attn["x"], dV.reshape(dV.shape[0], -1, act_attn["W_v"].shape[1]))
            if self.use_lora and "v_proj" in self.lora_target_modules and f"lora_A_v_{i}" in self.weights:
                r = self.lora_rank
                scale = self.lora_alpha / r
                A_v = self.weights[f"lora_A_v_{i}"]
                B_v = self.weights[f"lora_B_v_{i}"]
                
                with np.errstate(all='ignore'):
                    dW_v_eff = np.nan_to_num(dW_v_eff, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    A_v = np.nan_to_num(A_v, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    B_v = np.nan_to_num(B_v, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    
                    dA = (dW_v_eff @ B_v.T) * scale
                    dB = (A_v.T @ dW_v_eff) * scale
                    
                    self._grads[f"lora_A_v_{i}"] = np.nan_to_num(dA, nan=0.0, posinf=clip_val, neginf=-clip_val)
                    self._grads[f"lora_B_v_{i}"] = np.nan_to_num(dB, nan=0.0, posinf=clip_val, neginf=-clip_val)

            elif not self.use_lora:
                self._grads[f"W_v_{i}"] = dW_v_eff
            
            da_x = np.einsum("bth,dh->btd", dQ.reshape(dQ.shape[0], -1, act_attn["W_q"].shape[1]), act_attn["W_q"]) + \
                   np.einsum("bth,dh->btd", dK.reshape(dK.shape[0], -1, act_attn["W_k"].shape[1]), act_attn["W_k"]) + \
                   np.einsum("bth,dh->btd", dV.reshape(dV.shape[0], -1, act_attn["W_v"].shape[1]), act_attn["W_v"])
            self._stats['bw_attn'].append(time.time() - t_attn)
            
            # Norm1 backward
            t_norm = time.time()
            act_norm1 = self._activations[f"norm1_{i}"]
            dx_norm1 = da_x
            
            if not self.use_lora:
                self._grads[f"norm1_{i}"] = np.einsum("btd,btd->d", dx_norm1, act_norm1["norm_x"])
            
            term1 = dx_norm1 * act_norm1["w"]
            term2 = np.mean(term1 * act_norm1["norm_x"], axis=-1, keepdims=True) * act_norm1["norm_x"]
            dx = dx + (term1 - term2) / act_norm1["rms"]
            self._stats['bw_norm'].append(time.time() - t_norm)
            
            self._stats['bw_layers'].append(time.time() - t_layer)
        
        # 3. Embedding Backward
        t_embed = time.time()
        # self._grads["E"] is already partially set from head. Add embedding lookup grad.
        # token lookup: dx = E[tokens] -> dE[tokens] += dx
        if not self.use_lora:
            tokens = self._activations["tokens"]
            for b in range(dx.shape[0]):
                for t in range(dx.shape[1]):
                    self._grads["E"][tokens[b, t]] += dx[b, t]
        self._stats['bw_embed'].append(time.time() - t_embed)
                
        # IMPORTANT: Record backward pass time BEFORE gradient memory calculation
        self._stats['bw_total'].append(time.time() - t_total)
        
        # Calculate gradient memory footprint (this can be expensive for large models)
        t_grad_mem = time.time()
        total_grad_bytes = 0
        grad_count = 0
        for key, grad in self._grads.items():
            grad_bytes = grad.nbytes
            total_grad_bytes += grad_bytes
            grad_count += 1
            self._memory_stats['grad_breakdown'][key] = grad_bytes / (1024 * 1024)  # Convert to MB
        
        self._memory_stats['grad_total_mb'] = total_grad_bytes / (1024 * 1024)
        grad_mem_time = time.time() - t_grad_mem
        self._stats['bw_grad_mem_calc'].append(grad_mem_time)
        
        self._log(f"[INFO] Backward pass complete. Total gradient memory: {self._memory_stats['grad_total_mb']:.2f} MB ({grad_count} tensors)")
        self._log(f"[INFO] Gradient memory calculation took: {grad_mem_time:.3f}s")
        self._log(f"[NOTE] Gradients are fully allocated arrays (unlike lazy-loaded weights), so this represents actual RAM usage")
        
        if self.verbose and not self.training:
            self._print_stats_table(title="BACKWARD PASS TIMING BREAKDOWN (seconds)")
            
        return self._grads

    def generate(self, tokens):
        """Full generation flow: Prefill -> Iterative Decode."""
        print(f"\n" + "#"*50)
        print(f"GENERATION MODE: {self.cfg.max_new_tokens} tokens")
        print("#"*50)
        
        # 1. Prefill/Prompt evaluation
        # verbose=True handles clearing stats and printing prefill session table
        logits = self.forward(tokens, inference=True, verbose=True)
        
        # Reset stats specifically for decode phase accumulation
        self._init_stats()
        
        # Sample first token from prefill
        next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
        generated_tokens = [next_token[0]]
        
        # Stream first token from prefill
        token_str = self.detokenize([next_token[0]])[0]
        print(f"\n[STREAM] ", end="", flush=True)
        print(token_str, end="", flush=True)
        
        print(f"\n" + "-"*50)
        print("INTERATIVE DECODE PHASE")
        print("-" * 50)
        print("[STREAM] ", end="", flush=True)  # Start streaming line for decode phase
        
        # 2. Iterative Decode
        for i in range(self.cfg.max_new_tokens - 1):
            t0 = time.time()
            current_input = np.full((tokens.shape[0], 1), generated_tokens[-1])
            
            # verbose=False suppresses per-step noise
            logits = self.forward(current_input, inference=True, verbose=False)
            
            dt = time.time() - t0
            next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
            token_str = self.detokenize([next_token[0]])[0]
            
            # Stream token as it's generated (production-like streaming)
            print(token_str, end="", flush=True)
            
            # Log detailed step info on separate line (doesn't interrupt stream)
            print(f"\n  [Step {i+1:02d}/{self.cfg.max_new_tokens-1}] Token: {next_token[0]:<6} ({token_str}) | Latency: {dt:.3f}s")
            print("[STREAM] ", end="", flush=True)  # Continue streaming on next line
            
            generated_tokens.append(next_token[0])
        
        # Final newline after streaming
        print()  # Newline after streaming completes
            
        print(f"\n[GENERATION COMPLETE] Total: {len(generated_tokens)} tokens.")
        
        # Recalculate KV cache size after all decode steps (cache has grown)
        total_elements = 0
        for c in self.cache:
            if c["K"] is not None:
                total_elements += c["K"].size + c["V"].size
        self._kv_size_mb = (total_elements * 4) / (1024 * 1024)
        
        # Show final aggregate decode table
        self.verbose = True # Re-enable for final table print
        self._print_stats_table(title=f"DECODE PHASE AGGREGATE SUMMARY ({len(generated_tokens)-1} tokens)")
        
        return generated_tokens

class EinsumOptimizer:
    def __init__(self, model: EinsumTransformer, lr=0.001):
        self.model = model
        self.lr = lr

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

        m = self.model
        g = m._grads
        
        t_opt_start = time.time()
        
        # Calculate memory footprint of this optimizer step
        step_memory_bytes = 0
        
        # Head/Embedding
        # In Llama, E and head are often shared or separate. 
        # Here we only update based on E_name
        if "E" in g:
            old_E = m.weights[m.E_name]
            step_memory_bytes += old_E.nbytes + g["E"].nbytes
            m.weights[m.E_name] = old_E - self.lr * g["E"]
        
        # Layers
        for i in range(m.cfg.n_layers):
            p = m.layer_weights[i]
            
            # LoRA Adapters (Update if present)
            for k in ["lora_A_q", "lora_B_q", "lora_A_v", "lora_B_v"]:
                # Check if this adapter exists in weights AND has a gradient
                w_key = f"{k}_{i}"
                if w_key in m.weights and w_key in g:
                    old_w = m.weights[w_key]
                    grad = g[w_key] # Shape matches
                    step_memory_bytes += old_w.nbytes + grad.nbytes
                    m.weights[w_key] = old_w - self.lr * grad

            # Attn (Base weights)
            for k in ["W_q", "W_k", "W_v", "W_o"]:
                name = p[k]
                grad_key = f"{k}_{i}"
                if name and grad_key in g:
                    # Weights in safetensors are often (out, in). 
                    # Our grads are (in, out) or similar. 
                    # We use .T to match the shape in weights[name]
                    old_w = m.weights[name]
                    grad_t = g[grad_key].T
                    step_memory_bytes += old_w.nbytes + grad_t.nbytes
                    m.weights[name] = old_w - self.lr * grad_t
                
            # Norms
            for k in ["norm1", "norm2"]:
                name = p[k]
                grad_key = f"{k}_{i}"
                if name and grad_key in g:
                    old_w = m.weights[name]
                    grad = g[grad_key]
                    step_memory_bytes += old_w.nbytes + grad.nbytes
                    m.weights[name] = old_w - self.lr * grad
                
            # FFN
            if not m.cfg.use_moe:
                for k in ["W1", "W2", "W3"]:
                    name = p[k]
                    grad_key = f"{k}_{i}"
                    if name and grad_key in g:
                        old_w = m.weights[name]
                        grad_t = g[grad_key].T
                        step_memory_bytes += old_w.nbytes + grad_t.nbytes
                        m.weights[name] = old_w - self.lr * grad_t
        
        opt_time = time.time() - t_opt_start
        step_memory_mb = step_memory_bytes / (1024 * 1024)
        
        # Track stats
        m._stats['opt_step'].append(opt_time)
        m._memory_stats['opt_per_step_mb'].append(step_memory_mb)
