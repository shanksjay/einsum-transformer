import re
import numpy as np
import os
import time
import copy
from pathlib import Path
from collections import OrderedDict
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

        self.d_head = self.d_model // self.n_heads


class LazyWeightManager:
    def __init__(self, weight_paths, target_dtype="float32", max_cache_size=512):
        self.handles = {}
        self.tensor_to_shard = {}
        self.target_dtype = target_dtype
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()
        
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
            
        self.cache[key] = res
        # Evict if necessary
        if self.max_cache_size and len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        return res

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
        if self.max_cache_size and len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)

    def purge_cache(self):
        print(f"[INFO] Purging weight cache ({len(self.cache)} tensors removed)")
        self.cache.clear()

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

    def purge_cache(self):
        pass


class EinsumTransformer:
    def __init__(self, cfg: TransformerConfig, *, _from_weights=None, _E_name=None, _layer_weights=None):
        self.cfg = cfg
        self.weights = None  # LazyWeightManager or DictWeightManager
        self.cache = [{"K": None, "V": None} for _ in range(cfg.n_layers)]
        self.cur_pos = 0
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
            print(f"[INFO] Mixed-precision quantization: {self.cfg.quantization} (W{getattr(self.cfg,'quantization_weight_bits',0)} A{getattr(self.cfg,'quantization_activation_bits',0)})")

    def clear_cache(self):
        self.cache = [{"K": None, "V": None} for _ in range(self.cfg.n_layers)]
        self.cur_pos = 0

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
        V, d, d_ff = c.vocab_size, c.d_model, c.d_ff
        B, T = c.batch_size, c.seq_len
        n_layers = c.n_layers
        d_kv = c.n_kv_heads * (c.d_model // c.n_heads)

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
        bw = (4 - (1 if wb == 8 else 0.5 if wb == 4 else 4)) if wb in (4, 8) else 0
        ba = (4 - (2 if ab == 16 else 1 if ab == 8 else 0.5 if ab == 4 else 4)) if ab in (4, 8, 16) else 0

        emb_mb = mb(V * d * bw + ((B * T * d * ba) if ba else 0))
        rms_mb = mb(2 * d * n_layers * bw) if bw else 0
        attn_pl = (2 * d * d + 2 * d_kv * d) * bw + (B * T * d * ba if ba else 0)
        attn_mb = mb(attn_pl * n_layers)
        ffn_pl = 3 * d * d_ff * bw + (B * T * d * ba if ba else 0)
        ffn_mb = mb(ffn_pl * n_layers)

        def fmt_mb(x): return f"{x:.3f} MB" if x > 0 else "—"

        print(f"[INFO] Per-layer operation breakdown (by tensor datatype, source→target) and memory saving:")
        print(f"  {'Operation':<18} {'Weights':<18} {'Activations':<18} {'Mem save':<14}")
        print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*14}")
        print(f"  {'Embedding':<18} {weight_str():<18} {act_str():<18} {fmt_mb(emb_mb):<14}")
        print(f"  {'RMSNorm (×2)':<18} {weight_str():<18} {'float32':<18} {fmt_mb(rms_mb):<14}")
        print(f"  {'Attn (Q,K,V,O)':<18} {weight_str():<18} {act_str():<18} {fmt_mb(attn_mb):<14}")
        print(f"  {'FFN (W1,W2,W3)':<18} {weight_str():<18} {act_str():<18} {fmt_mb(ffn_mb):<14}")
        if self.use_lora:
            print(f"  {'LoRA (A,B)':<18} {'float32':<18} {'-':<18} {'—':<14}")
        print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*14}")
        print(f"  {'Total (all layers)':<18} {'':<18} {'':<18} {fmt_mb(emb_mb + rms_mb + attn_mb + ffn_mb):<14}")

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

    def rms_norm(self, x, w, eps=1e-6, key=None):
        t0 = time.time()
        # Use higher precision for RMS calculation to avoid overflow
        x_fp64 = x.astype(np.float64)
        rms = np.sqrt(np.mean(x_fp64**2, axis=-1, keepdims=True) + eps).astype(x.dtype)
        # Handle cases where rms might be zero or non-finite
        rms = np.where(np.isfinite(rms) & (rms > 0), rms, eps)
        norm_x = x / rms
        res = norm_x * w
        if hasattr(self, '_stats') and 'norm' in self._stats: self._stats['norm'].append(time.time() - t0)
        if self.training and key is not None: self._activations[key] = {"x": x, "rms": rms, "norm_x": norm_x}
        return res

    def swiglu(self, x, W1, W2, W3, key=None):
        t0 = time.time()
        a = np.einsum("btd,df->btf", x, W1, optimize=True)
        b = np.einsum("btd,df->btf", x, W2, optimize=True)
        sig_b = 1.0 / (1.0 + np.exp(-b))
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
        t_load = time.time()
        W_q, W_k, W_v, W_o = [self.get_w(p[k], transpose=True) for k in ["W_q", "W_k", "W_v", "W_o"]]
        lora_q = lora_v = None
        if self.use_lora:
            scale = self.lora_alpha / self.lora_rank
            if p.get("lora_A_q") is not None:
                lora_q = (self.get_w(p["lora_A_q"]), self.get_w(p["lora_B_q"]), scale)
            if p.get("lora_A_v") is not None:
                lora_v = (self.get_w(p["lora_A_v"]), self.get_w(p["lora_B_v"]), scale)
        self._stats['attn_load'].append(time.time() - t_load)
        
        t_qkv = time.time()
        B, T = x.shape[0], x.shape[1]
        Q_proj = np.einsum("btd,dh->bth", x, W_q, optimize=True)
        if lora_q:
            A, Bl, s = lora_q
            xA = np.einsum("btd,dr->btr", x, A, optimize=True)
            Q_proj += s * np.einsum("btr,rh->bth", xA, Bl, optimize=True)
            if self.training: self._activations[f"lora_q_{layer_idx}"] = {"xA": xA, "A": A, "B": Bl, "s": s}
        Q = Q_proj.reshape(B, T, c.n_heads, c.d_head)
        K = np.einsum("btd,dh->bth", x, W_k, optimize=True).reshape(B, T, c.n_kv_heads, c.d_head)
        V_proj = np.einsum("btd,dh->bth", x, W_v, optimize=True)
        if lora_v:
            A, Bl, s = lora_v
            xA = np.einsum("btd,dr->btr", x, A, optimize=True)
            V_proj += s * np.einsum("btr,rh->bth", xA, Bl, optimize=True)
            if self.training: self._activations[f"lora_v_{layer_idx}"] = {"xA": xA, "A": A, "B": Bl, "s": s}
        V = V_proj.reshape(B, T, c.n_kv_heads, c.d_head)
        self._stats['attn_qkv'].append(time.time() - t_qkv)

        if c.use_mqa or c.use_gqa:
            repeats = c.n_heads // c.n_kv_heads
            K = np.repeat(K, repeats, axis=2)
            V = np.repeat(V, repeats, axis=2)

        t_rope = time.time()
        pos = np.arange(self.cur_pos, self.cur_pos + T) if inference else np.arange(T)
        Q, K = self.apply_rope(Q, pos), self.apply_rope(K, pos)
        self._stats['attn_rope'].append(time.time() - t_rope)

        Q_t, K_t, V_t = [t.transpose(0, 2, 1, 3) for t in [Q, K, V]]
        if inference:
            if self.cache[layer_idx]["K"] is None: self.cache[layer_idx]["K"], self.cache[layer_idx]["V"] = K_t, V_t
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
        weighted_v = np.einsum("bhij,bhjd->bhid", attn, V_t, optimize=True)
        weighted_v_merged = weighted_v.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
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
        for e in range(self.cfg.num_experts):
            b_idx, t_idx, k_idx = np.where(top_indices == e)
            if len(b_idx) == 0: continue
            W1, W2, W3 = [self.get_w(p["experts"][e][k], transpose=True) for k in ["W1", "W2", "W3"]]
            res_expert = self.swiglu(x[b_idx, t_idx][None, :, :], W1, W2, W3)
            out[b_idx, t_idx] += res_expert[0]
        self._stats['ffn'].append(time.time() - t0)
        return out / self.cfg.top_k

    def forward(self, tokens, inference=False, verbose=True, training=False, max_layers=None):
        self.verbose, self.training = verbose, training
        if not hasattr(self, '_stats'): self._init_stats()
        if training: self._activations, self._grads = {"tokens": tokens}, {}
        is_decode = inference and any(c["K"] is not None for c in self.cache)
        if inference and not is_decode: self.cur_pos = 0
        t_stage_start = time.time()
        t_embed = time.time()
        E = self.get_w(self.E_name)
        x = E[tokens]
        if 'embedding' not in self._stats: self._stats['embedding'] = []
        self._stats['embedding'].append(time.time() - t_embed)
        ab = getattr(self.cfg, "quantization_activation_bits", 0)
        if ab in (4, 8, 16): x = self._quantize_dequant_act(x.astype(np.float32), ab)
        if training: self._activations["embedding"] = x
        
        n_layers = max_layers if max_layers is not None else self.cfg.n_layers
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
            self._stats['layers'].append(time.time() - t_layer_start)

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
            self._memory_stats['fwd_activation_mb'] = _sum_bytes(self._activations) / (1024 * 1024)

        t_head = time.time()
        res = np.einsum("btd,vd->btv", x, E, optimize=True)
        if 'output_head' not in self._stats: self._stats['output_head'] = []
        self._stats['output_head'].append(time.time() - t_head)
        if self.verbose and not self.training and not is_decode: self._print_stats_table()
        return res

    def backward(self, d_logits):
        t_total = time.time()
        c = self.cfg
        E = self.get_w(self.E_name)
        final_x = self._activations["final_x"]
        dx = np.einsum("btv,vd->btd", d_logits, E, optimize=True)
        if not self.use_lora: self._grads["E"] = np.einsum("btv,btd->vd", d_logits, final_x, optimize=True)
        for i in reversed(range(c.n_layers)):
            p = self.layer_weights[i]
            if not c.use_moe:
                act_ffn, dffn_out = self._activations[f"ffn_{i}"], dx
                W1, W2, W3 = [self.get_w(p[k], transpose=True) for k in ["W1", "W2", "W3"]]
                if not self.use_lora: self._grads[f"W3_{i}"] = np.einsum("btf,btd->fd", act_ffn["gated"], dffn_out, optimize=True)
                dgated = np.einsum("btd,fd->btf", dffn_out, W3, optimize=True)
                da = dgated * (act_ffn["b"] * act_ffn["sig_b"])
                db = dgated * act_ffn["a"] * (act_ffn["sig_b"] * (1 + act_ffn["b"] * (1 - act_ffn["sig_b"])))
                if not self.use_lora:
                    self._grads[f"W1_{i}"], self._grads[f"W2_{i}"] = np.einsum("btd,btf->df", act_ffn["x"], da, optimize=True), np.einsum("btd,btf->df", act_ffn["x"], db, optimize=True)
                dx = dx + np.einsum("btf,df->btd", da, W1, optimize=True) + np.einsum("btf,df->btd", db, W2, optimize=True)
                act_norm2, dx_norm2 = self._activations[f"norm2_{i}"], dx
                norm2_w = self.get_w(p["norm2"])
                if not self.use_lora: self._grads[f"norm2_{i}"] = np.einsum("btd,btd->d", dx_norm2, act_norm2["norm_x"], optimize=True)
                term1 = dx_norm2 * norm2_w
                dx = dx + (term1 - np.mean(term1 * act_norm2["norm_x"], axis=-1, keepdims=True) * act_norm2["norm_x"]) / act_norm2["rms"]
            act_attn, dattn_out = self._activations[f"attn_{i}"], dx
            W_o = self.get_w(p["W_o"], transpose=True)
            if not self.use_lora: self._grads[f"W_o_{i}"] = np.einsum("btd,bth->dh", act_attn["weighted_v_merged"], dattn_out, optimize=True)
            dweighted = np.einsum("bth,dh->btd", dattn_out, W_o, optimize=True).reshape(dx.shape[0], dx.shape[1], c.n_heads, c.d_head).transpose(0, 2, 1, 3)
            dattn = np.einsum("bhid,bhjd->bhij", dweighted, act_attn["V_t"], optimize=True)
            dscores = act_attn["attn"] * (dattn - np.sum(dattn * act_attn["attn"], axis=-1, keepdims=True)) / np.sqrt(c.d_head)
            dQ_t, dK_t_f = np.einsum("bhij,bhjd->bhid", dscores, act_attn["K_t"], optimize=True), np.einsum("bhji,bhjd->bhid", dscores, act_attn["Q_t"], optimize=True)
            dV_t_f = np.einsum("bhij,bhid->bhjd", act_attn["attn"], dweighted, optimize=True)
            groups = c.n_heads // c.n_kv_heads
            dQ, dK = dQ_t.transpose(0, 2, 1, 3), dK_t_f.reshape(dx.shape[0], c.n_kv_heads, groups, -1, c.d_head).sum(2).transpose(0, 2, 1, 3)
            dV = dV_t_f.reshape(dx.shape[0], c.n_kv_heads, groups, -1, c.d_head).sum(2).transpose(0, 2, 1, 3)
            dQ, dK = self._rope_backward(dQ, np.arange(dQ.shape[1])), self._rope_backward(dK, np.arange(dK.shape[1]))
            dQf, dKf, dVf = [np.clip(t.reshape(dx.shape[0], dx.shape[1], -1), -1.0, 1.0) for t in [dQ, dK, dV]]
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
                self._grads[f"W_q_{i}"], self._grads[f"W_k_{i}"], self._grads[f"W_v_{i}"] = [np.einsum("btd,bth->dh", act_attn["x"], t, optimize=True) for t in [dQf, dKf, dVf]]
            W_q, W_k, W_v = [self.get_w(p[k], transpose=True) for k in ["W_q", "W_k", "W_v"]]
            dx = dx + np.einsum("bth,dh->btd", dQf, W_q, optimize=True) + np.einsum("bth,dh->btd", dKf, W_k, optimize=True) + np.einsum("bth,dh->btd", dVf, W_v, optimize=True)
            if self.use_lora:
                if dQ_B is not None: dx += self._activations[f"lora_q_{i}"]["s"] * np.einsum("btr,dr->btd", dQ_B, self._activations[f"lora_q_{i}"]["A"], optimize=True)
                if dV_B is not None: dx += self._activations[f"lora_v_{i}"]["s"] * np.einsum("btr,dr->btd", dV_B, self._activations[f"lora_v_{i}"]["A"], optimize=True)
            act_norm1, dx_norm1 = self._activations[f"norm1_{i}"], dx
            norm1_w = self.get_w(p["norm1"])
            if not self.use_lora: self._grads[f"norm1_{i}"] = np.einsum("btd,btd->d", dx_norm1, act_norm1["norm_x"], optimize=True)
            term1 = dx_norm1 * norm1_w
            dx = dx + (term1 - np.mean(term1 * act_norm1["norm_x"], axis=-1, keepdims=True) * act_norm1["norm_x"]) / act_norm1["rms"]
        if not self.use_lora: np.add.at(self._grads["E"], self._activations["tokens"], dx)

        # Calculate gradient memory footprint
        total_grad_bytes = sum(g.nbytes for g in self._grads.values() if isinstance(g, np.ndarray))
        self._memory_stats['grad_total_mb'] = total_grad_bytes / (1024 * 1024)

        self._stats['bw_total'].append(time.time() - t_total)
        return self._grads

    def generate(self, tokens):
        if getattr(self.cfg, 'speculative', False): return self.speculative_generate(tokens)
        self.clear_cache()
        logits = self.forward(tokens, inference=True, verbose=True)
        next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
        generated = [next_token[0]]
        print(f"\n[STREAM] {self.detokenize([next_token[0]])[0]}", end="", flush=True)
        for i in range(self.cfg.max_new_tokens - 1):
            logits = self.forward(np.full((tokens.shape[0], 1), generated[-1]), inference=True, verbose=False)
            next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
            generated.append(next_token[0])
            print(self.detokenize([next_token[0]])[0], end="", flush=True)
        print()
        return generated

    def speculative_generate(self, tokens, k=4):
        print(f"[INFO] Starting speculative decoding (k={k}, draft_layers={self.cfg.draft_layers})...")
        self.clear_cache()
        draft_cfg = copy.deepcopy(self.cfg)
        draft_cfg.n_layers = self.cfg.draft_layers
        draft_model = EinsumTransformer(draft_cfg, _from_weights=self.weights,
                                        _E_name=self.E_name,
                                        _layer_weights=self.layer_weights[:self.cfg.draft_layers])
        
        # 1. Prefill
        logits = self.forward(tokens, inference=True, verbose=True)
        _ = draft_model.forward(tokens, inference=True, verbose=False)
        
        # 2. Sample first token
        next_token = self.sample_top_k(logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)
        generated = [next_token[0]]
        print(f"\n[STREAM] {self.detokenize([next_token[0]])[0]}", end="", flush=True)
        
        # 3. Speculative loop
        while len(generated) < self.cfg.max_new_tokens:
            # a. Draft predicts k tokens
            d_tokens = []
            draft_input = np.full((1, 1), generated[-1])
            for _ in range(k):
                d_logits = draft_model.forward(draft_input, inference=True, verbose=False)
                d_tok = self.sample_top_k(d_logits, k=self.cfg.top_k_sample, temperature=self.cfg.temperature)[0]
                d_tokens.append(d_tok)
                draft_input = np.full((1, 1), d_tok)
            
            # b. Target verifies in one batch forward pass
            # Feed the last confirmed token plus all but the last draft prediction
            v_input = np.array([generated[-1]] + d_tokens[:-1]).reshape(1, -1)
            v_logits = self.forward(v_input, inference=True, verbose=False)
            
            # c. Verify
            n_accepted = 0
            for j in range(k):
                target_tok = self.sample_top_k(v_logits[:, j:j+1, :], k=self.cfg.top_k_sample, temperature=self.cfg.temperature)[0]
                if target_tok == d_tokens[j]:
                    generated.append(d_tokens[j])
                    print(self.detokenize([d_tokens[j]])[0], end="", flush=True)
                    n_accepted += 1
                else:
                    generated.append(target_tok)
                    print(f"[{self.detokenize([target_tok])[0]}]", end="", flush=True)
                    break
            
            # d. Rollback target model cache to n_accepted confirmed tokens
            n_target_rollback = k - n_accepted
            if n_target_rollback > 0:
                for layer in self.cache:
                    if layer["K"] is not None:
                        layer["K"] = layer["K"][:, :, :-n_target_rollback, :]
                        layer["V"] = layer["V"][:, :, :-n_target_rollback, :]
                self.cur_pos -= n_target_rollback

            # e. Sync draft model cache to target model's confirmed state
            for l in range(self.cfg.draft_layers):
                draft_model.cache[l]["K"] = self.cache[l]["K"].copy()
                draft_model.cache[l]["V"] = self.cache[l]["V"].copy()
            draft_model.cur_pos = self.cur_pos

        print()
        return generated

    def _init_stats(self):
        self._stats = {k: [] for k in ['attn', 'ffn', 'layers', 'norm', 'weight_load', 'attn_load', 'attn_qkv', 'attn_rope', 'attn_score', 'attn_out', 'ffn_load', 'ffn_compute', 'prefill', 'decode', 'bw_total', 'bw_layers', 'bw_attn', 'bw_ffn', 'bw_norm', 'bw_head', 'bw_embed', 'opt_step', 'embedding', 'output_head']}
        self._memory_stats = {'grad_total_mb': 0.0, 'fwd_activation_mb': 0.0, 'grad_breakdown': {}, 'opt_per_step_mb': []}

    def _rope_backward(self, dy, positions):
        B, T, H, D = dy.shape
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
        self._log(title)
        self._log("-" * 50)
        self._log(f"{'Phase':<20} | {'p50':<10} | {'p99':<10}")
        self._log("-" * 50)
        def pr(n, ts):
            if ts: self._log(f"{n:<20} | {np.percentile(ts, 50):<10.4f} | {np.percentile(ts, 99):<10.4f}")
        is_bw = "BACKWARD" in title.upper()
        if not is_bw:
            pr("Embedding", self._stats.get('embedding'))
            pr("Layer Total", self._stats.get('layers'))
            pr("  Norm", self._stats.get('norm'))
            pr("  Weight Load", self._stats.get('weight_load'))
            pr("  Attention", self._stats.get('attn'))
            pr("  FeedForward", self._stats.get('ffn'))
            pr("    Compute", self._stats.get('ffn_compute'))
            pr("Output Head", self._stats.get('output_head'))
            if self._stats.get('prefill'): pr("Prefill Stage", self._stats['prefill'])
            if self._stats.get('decode'): pr("Decode Stage", self._stats['decode'])
        else:
            pr("Backward Pass", self._stats.get('bw_total'))
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
        m, g, lr = self.model, self.model._grads, self.lr
        t0 = time.time()

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

        step_memory_bytes = 0
        if "E" in g:
            step_memory_bytes += m.weights[m.E_name].nbytes + g["E"].nbytes
            m.weights[m.E_name] -= lr * g["E"]
        for i in range(m.cfg.n_layers):
            p = m.layer_weights[i]
            for k in ["lora_A_q", "lora_B_q", "lora_A_v", "lora_B_v"]:
                w_key = f"{k}_{i}"
                if w_key in m.weights and w_key in g:
                    step_memory_bytes += m.weights[w_key].nbytes + g[w_key].nbytes
                    m.weights[w_key] -= lr * g[w_key]
            for k in ["W_q", "W_k", "W_v", "W_o", "W1", "W2", "W3"]:
                grad_key = f"{k}_{i}"
                if p.get(k) and grad_key in g:
                    step_memory_bytes += m.weights[p[k]].nbytes + g[grad_key].nbytes
                    m.weights[p[k]] -= lr * g[grad_key].T
            for k in ["norm1", "norm2"]:
                grad_key = f"{k}_{i}"
                if p.get(k) and grad_key in g:
                    step_memory_bytes += m.weights[p[k]].nbytes + g[grad_key].nbytes
                    m.weights[p[k]] -= lr * g[grad_key]

        m._memory_stats['opt_per_step_mb'].append(step_memory_bytes / (1024 * 1024))
        m._stats['opt_step'].append(time.time() - t0)
