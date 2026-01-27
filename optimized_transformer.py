import numpy as np
import time
import copy
from core_transformer import TransformerConfig

try:
    from scipy.special import expit
    USE_SCIPY = True
except ImportError:
    USE_SCIPY = False

class OptimizedTransformer:
    def __init__(self, cfg: TransformerConfig, parent_model=None, shared_weights=None):
        self.cfg = cfg
        # Increase max seq len to handle generation
        self.max_seq_len = max(cfg.seq_len, 4096)
        self.batch_size = cfg.batch_size
        self.cur_pos = 0

        # Pre-allocate KV cache: [Batch, Layers, 2(K,V), MaxLen, n_kv_heads, head_dim]
        # float32 to avoid cast overhead
        self.kv_cache = np.zeros(
            (self.batch_size, cfg.n_layers, 2, self.max_seq_len, cfg.n_kv_heads, cfg.d_head),
            dtype=np.float32
        )

        # Pre-compute RoPE frequencies
        self._init_rope()

        if shared_weights:
            self.params = shared_weights['params']
            self.layers = shared_weights['layers']
            if len(self.layers) < cfg.n_layers:
                 raise ValueError("Shared layers fewer than config n_layers")
            # Slice layers if this is a draft model
            self.layers = self.layers[:cfg.n_layers]
        elif parent_model:
            self._import_weights(parent_model)
        else:
            raise NotImplementedError("Direct loading not implemented, pass parent_model")

        self.draft_model = None

    def _init_rope(self):
        head_dim = self.cfg.d_head
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
        t = np.arange(self.max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = np.cos(emb).astype(np.float32)
        self.sin_cached = np.sin(emb).astype(np.float32)

    def _import_weights(self, model):
        c = self.cfg
        print("[INFO] Optimizing weights for inference (FUSED)...")

        self.params = {}
        self.params['E'] = model.get_w(model.E_name).astype(np.float32)

        self.layers = []
        for i in range(c.n_layers):
            p = model.layer_weights[i]
            layer_w = {}

            def load_linear(key, transpose=True):
                w = model.get_w(p[key], transpose=transpose)
                return np.ascontiguousarray(w)

            wq = load_linear('W_q')
            wk = load_linear('W_k')
            wv = load_linear('W_v')
            layer_w['W_qkv'] = np.ascontiguousarray(np.concatenate([wq, wk, wv], axis=1))

            layer_w['W_o'] = load_linear('W_o')
            layer_w['norm1'] = model.get_w(p['norm1'])
            layer_w['norm2'] = model.get_w(p['norm2'])

            if c.use_moe:
                raise NotImplementedError("MoE optimization not yet implemented")
            else:
                w1 = load_linear('W1')
                w2 = load_linear('W2')
                layer_w['W_gate_up'] = np.ascontiguousarray(np.concatenate([w1, w2], axis=1))
                layer_w['W3'] = load_linear('W3')

            self.layers.append(layer_w)

        # EinsumTransformer uses tied weights (uses E for output projection)
        self.params['Head'] = self.params['E'].T

    def clear_cache(self):
        self.cur_pos = 0
        self.kv_cache.fill(0)

    def rms_norm(self, x, w, eps=1e-6):
        x_fp32 = x.astype(np.float32)
        variance = np.mean(x_fp32**2, axis=-1, keepdims=True)
        rms = np.sqrt(variance + eps)
        return (x_fp32 / rms) * w

    def forward_chunk(self, tokens):
        """
        Unified forward pass for prefill, decode, and verification.
        tokens: [B, L]
        """
        B, L = tokens.shape
        c = self.cfg
        start_pos = self.cur_pos

        if B > self.batch_size:
            raise ValueError(f"Batch size {B} exceeds allocated {self.batch_size}")

        x = np.take(self.params['E'], tokens, axis=0) # [B, L, D]

        d_q = c.n_heads * c.d_head
        d_kv = c.n_kv_heads * c.d_head
        d_gate = c.d_ff

        # RoPE Cache Slice
        # [L, D]
        cos_T = self.cos_cached[start_pos : start_pos+L]
        sin_T = self.sin_cached[start_pos : start_pos+L]

        # If L=1, reshape for broadcasting [1, D] -> [1, 1, 1, D] inside loop
        if L == 1:
            cos_T = cos_T[None, None, :, :] # [1, 1, 1, D] (expanded later)
            sin_T = sin_T[None, None, :, :]
        else:
            cos_T = cos_T[None, :, None, :] # [1, L, 1, D]
            sin_T = sin_T[None, :, None, :]

        # Causal Mask for L > 1
        mask = None
        if L > 1:
            mask = np.triu(np.ones((L, L), dtype=np.float32), k=1) * -1e9

        for i, l in enumerate(self.layers):
            h = self.rms_norm(x, l['norm1'])

            # QKV
            qkv = h @ l['W_qkv'] # [B, L, D_q + 2*D_kv]

            q = qkv[..., :d_q]
            k = qkv[..., d_q:d_q+d_kv]
            v = qkv[..., d_q+d_kv:]

            q = q.reshape(B, L, c.n_heads, c.d_head)
            k = k.reshape(B, L, c.n_kv_heads, c.d_head)
            v = v.reshape(B, L, c.n_kv_heads, c.d_head)

            # RoPE
            # For L=1, broadcast works. For L>1, broadcast works.
            # cos_T: [1, L, 1, D]. q: [B, L, H, D].
            # We need to unsqueeze cos/sin to match H dim if needed?
            # cos_T is [1, L, 1, D]. Broadcasts to [B, L, H, D]. Correct.

            # Pre-slice for RoPE
            # Since q/k are contiguous, this is fast.
            # Standard:
            q_r, q_i = q[..., 0::2], q[..., 1::2]
            k_r, k_i = k[..., 0::2], k[..., 1::2]

            # Broadcasting slices of cos/sin
            c_half = cos_T[..., 0::2]
            s_half = sin_T[..., 0::2]

            # In-place update
            # We use temporaries to ensure correctness
            q_r_new = q_r * c_half - q_i * s_half
            q_i_new = q_r * s_half + q_i * c_half
            q[..., 0::2] = q_r_new
            q[..., 1::2] = q_i_new

            k_r_new = k_r * c_half - k_i * s_half
            k_i_new = k_r * s_half + k_i * c_half
            k[..., 0::2] = k_r_new
            k[..., 1::2] = k_i_new

            # Update Cache
            self.kv_cache[:B, i, 0, start_pos:start_pos+L, :, :] = k
            self.kv_cache[:B, i, 1, start_pos:start_pos+L, :, :] = v

            # Attention
            # Q: [B, L, H, D]
            # K_cache: [B, start_pos+L, H_kv, D] (Need to read full history)
            # But wait, self.kv_cache shape is [B, L_model, 2, MaxLen, H, D].
            # slice: self.kv_cache[:B, i, 0, :start_pos+L, :, :] -> [B, T_full, H, D]

            k_cache = self.kv_cache[:B, i, 0, :start_pos+L, :, :]
            v_cache = self.kv_cache[:B, i, 1, :start_pos+L, :, :]

            # Transpose K -> [B, H, D, T_full]
            # Handle GQA
            if c.n_heads != c.n_kv_heads:
                n_rep = c.n_heads // c.n_kv_heads
                # Expand Q: [B, L, H_kv, Rep, D]
                q_reshaped = q.reshape(B, L, c.n_kv_heads, n_rep, c.d_head).transpose(0, 2, 1, 3, 4) # [B, Hkv, L, Rep, D]

                # K: [B, T, Hkv, D] -> [B, Hkv, D, T]
                k_t = k_cache.transpose(0, 2, 3, 1)
                # Reshape for broadcasting against Q [B, Hkv, L, Rep, D]
                # We want K to look like [B, Hkv, 1, D, T] so stack dims (B,Hkv,1) broadcast with (B,Hkv,L)
                k_t = k_t[:, :, None, :, :]

                # Score: [B, Hkv, L, Rep, D] @ [B, Hkv, 1, D, T] -> [B, Hkv, L, Rep, T]
                # If L=1 (Decode), Q is [B, Hkv, 1, Rep, D].
                # If L>1 (Prefill), Q is [B, Hkv, L, Rep, D].

                scores = np.matmul(q_reshaped, k_t) / np.sqrt(c.d_head) # [B, Hkv, L, Rep, T]

                # Apply Mask
                if mask is not None:
                    # mask is [L, L]. scores last dim is T (start_pos+L).
                    # We only mask the interaction between new tokens (last L x last L block).
                    # history (0..start_pos) is fully visible.
                    # Broadcast mask: [1, 1, L, 1, T] ? No.
                    # Simplest: padded mask [L, T].
                    # full_mask = [L, T] where last LxL is causal, rest is 0 (visible).
                    # But actually `scores` is [..., L, ..., T].
                    # `scores[..., :, :, :start_pos]` is fine.
                    # `scores[..., :, :, start_pos:]` needs mask.
                    scores[..., :, :, start_pos:] += mask[:, None, :] # Broadcast L, L to L, Rep, L?
                    # shape mismatch likely.
                    # mask is (L,L).
                    # scores is (..., L, Rep, T).
                    # target slice is (..., L, Rep, L).
                    # mask needs reshape to (L, 1, L).
                    scores[..., :, :, start_pos:] += mask.reshape(L, 1, L)

                max_score = np.max(scores, axis=-1, keepdims=True)
                exp_score = np.exp(scores - max_score)
                attn = exp_score / np.sum(exp_score, axis=-1, keepdims=True)

                # V: [B, T, Hkv, D] -> [B, Hkv, T, D]
                v_t = v_cache.transpose(0, 2, 1, 3)
                # Reshape V for broadcasting: [B, Hkv, 1, T, D]
                v_t = v_t[:, :, None, :, :]

                # Attn: [B, Hkv, L, Rep, T] @ [B, Hkv, 1, T, D] -> [B, Hkv, L, Rep, D]
                out = np.matmul(attn, v_t)

                # Reshape back to [B, L, H, D]
                out = out.transpose(0, 2, 1, 3, 4).reshape(B, L, c.d_model)

            else:
                # MHA
                # Q: [B, L, H, D] -> [B, H, L, D]
                q_t = q.transpose(0, 2, 1, 3)
                # K: [B, T, H, D] -> [B, H, D, T]
                k_t = k_cache.transpose(0, 2, 3, 1)

                scores = np.matmul(q_t, k_t) / np.sqrt(c.d_head) # [B, H, L, T]

                if mask is not None:
                    # mask [L, L]. scores [..., L, T].
                    scores[..., :, start_pos:] += mask

                max_score = np.max(scores, axis=-1, keepdims=True)
                exp_score = np.exp(scores - max_score)
                attn = exp_score / np.sum(exp_score, axis=-1, keepdims=True)

                # V: [B, T, H, D] -> [B, H, T, D]
                v_t = v_cache.transpose(0, 2, 1, 3)
                out = np.matmul(attn, v_t) # [B, H, L, D]
                out = out.transpose(0, 2, 1, 3).reshape(B, L, c.d_model)

            h = out @ l['W_o']
            x = x + h

            # FFN
            h = self.rms_norm(x, l['norm2'])
            gate_up = h @ l['W_gate_up']
            gate = gate_up[..., :d_gate]
            up = gate_up[..., d_gate:]

            if USE_SCIPY:
                # Scipy expit is sigmoid.
                # swish = x * sigmoid(x)
                # gate = gate * expit(gate) * up
                # expit is ufunc, fast C loop
                gate = gate * expit(gate) * up
            else:
                sig = np.where(gate > 0, 1.0 / (1.0 + np.exp(-gate)), np.exp(gate) / (1.0 + np.exp(gate)))
                gate = gate * sig * up

            down = gate @ l['W3']
            x = x + down

        logits = x @ self.params['Head']
        self.cur_pos += L
        return logits

    def generate(self, tokens, max_new_tokens):
        B, L = tokens.shape
        c = self.cfg
        self.clear_cache()

        # Prefill
        # print(f"[Optimized] Prefill {L} tokens...")
        self.forward_chunk(tokens)

        generated = tokens.tolist()

        if c.speculative and c.draft_layers < c.n_layers:
            return self.speculative_generate_loop(generated, max_new_tokens, B)

        # print(f"[Optimized] generating {max_new_tokens} tokens...")

        next_input = tokens[:, -1:] # [B, 1]

        for _ in range(max_new_tokens):
            logits = self.forward_chunk(next_input) # [B, 1, V]
            next_token = np.argmax(logits[:, -1, :], axis=-1) # [B]

            for b in range(B):
                generated[b].append(next_token[b])

            next_input = next_token[:, None] # [B, 1]

        return generated

    def _get_draft_model(self):
        if self.draft_model is None:
            # Create draft model sharing weights
            draft_cfg = copy.deepcopy(self.cfg)
            draft_cfg.n_layers = self.cfg.draft_layers
            draft_cfg.speculative = False

            shared = {'params': self.params, 'layers': self.layers}
            self.draft_model = OptimizedTransformer(draft_cfg, shared_weights=shared)
        return self.draft_model

    def speculative_generate_loop(self, generated, max_new_tokens, B):
        # generated is list of lists
        draft = self._get_draft_model()
        draft.clear_cache()

        # Sync draft cache with target cache (already prefilled)
        # Assuming layers match 0..draft_layers
        # Copy cache
        # draft.cur_pos should match self.cur_pos
        current_pos = self.cur_pos
        draft.cur_pos = current_pos

        # Copy the KV cache data
        # draft cache: [B, draft_L, 2, max_len, H, D]
        # self cache:  [B, n_L, 2, max_len, H, D]
        # Copy first draft_L layers
        n_d = self.cfg.draft_layers
        draft.kv_cache[:, :n_d, :, :current_pos, :, :] = self.kv_cache[:, :n_d, :, :current_pos, :, :]

        # K (speculative lookahead)
        K_spec = 4

        generated_count = 0

        next_input = np.array([g[-1] for g in generated])[:, None] # [B, 1]

        while generated_count < max_new_tokens:
            # 1. Draft Generate K tokens
            draft_tokens = []
            d_in = next_input

            # This loop is sequential and slow-ish, but draft is shallow
            for _ in range(K_spec):
                logits = draft.forward_chunk(d_in)
                tok = np.argmax(logits[:, -1, :], axis=-1)[:, None] # [B, 1]
                draft_tokens.append(tok)
                d_in = tok

            # draft_tokens is list of K arrays [B, 1]
            # Verify Input: [Current Last Token] + [K Draft Tokens]
            # We already ran target on Current Last Token during previous step or prefill?
            # NO. "next_input" was the last generated token.
            # We need to run target on next_input AND draft tokens.
            # Total K+1 tokens.

            # Wait, `next_input` was NOT processed by target yet?
            # In standard loop: forward(next_input) -> next_token.
            # Here: forward(next_input) is needed to verify the FIRST draft token?
            # No.
            # Draft predicts T1 from T0.
            # Target predicts T1' from T0.
            # We compare T1 and T1'.

            # So verification input sequence is: [T0, T1, T2, ..., Tk].
            # Target computes logits for T0 -> P0, T1 -> P1...
            # We verify: argmax(P0) == T1 ? argmax(P1) == T2 ?

            # Construct verification sequence
            # next_input is T0.
            # draft_tokens are T1, T2... Tk.

            # Stack: [B, K+1]
            # [T0, T1, ..., Tk]
            verify_input = np.concatenate([next_input] + draft_tokens, axis=1) # [B, K+1]

            # Run Target
            # This will advance target cache by K+1
            start_pos_before = self.cur_pos
            logits = self.forward_chunk(verify_input) # [B, K+1, V]

            # Verification Logic
            # We check each position j in [0..K-1]
            # Logit at j predicts token at j+1
            # We compare prediction with draft_tokens[j] (which is token j+1)

            # For B > 1, all must match to accept step? Or per-sequence?
            # To keep cache aligned, we usually do all-or-nothing or manage ragged cache.
            # "EinsumTransformer" does all-or-nothing.
            # Let's do all-or-nothing for simplicity.

            n_accepted = 0
            # Target predictions
            target_preds = np.argmax(logits, axis=-1) # [B, K+1]

            # draft_tokens[j] corresponds to verify_input[j+1]
            # target_preds[j] corresponds to prediction for verify_input[j+1]

            for j in range(K_spec):
                # draft token at step j (0-based) is draft_tokens[j]
                d_tok = draft_tokens[j] # [B, 1]
                t_pred = target_preds[:, j:j+1] # [B, 1]

                if np.array_equal(d_tok, t_pred):
                    n_accepted += 1
                else:
                    break

            # Append accepted tokens
            # If n_accepted = N, we accept N tokens.
            # The tokens are draft_tokens[:n_accepted]

            for j in range(n_accepted):
                toks = draft_tokens[j].flatten()
                for b in range(B):
                    generated[b].append(toks[b])

            generated_count += n_accepted

            # Bonus token?
            # If we accepted j tokens, we are at pos + j.
            # Target computed logits for pos+j (index j in logits).
            # This prediction is available (target_preds[:, j]).
            # We can always accept one "model" token at the end of the chain.

            bonus_tok = target_preds[:, n_accepted:n_accepted+1]
            for b in range(B):
                generated[b].append(bonus_tok[b, 0])
            generated_count += 1

            next_input = bonus_tok

            # Rollback
            # We processed K+1 tokens.
            # We keep: n_accepted (draft) + 1 (bonus) = n_accepted + 1.
            # We verify input size was K+1.
            # Cache advanced by K+1.
            # We want cache to advance by n_accepted + 1.
            # So rollback = (K+1) - (n_accepted + 1) = K - n_accepted.

            rollback = K_spec - n_accepted
            if rollback > 0:
                self.cur_pos -= rollback
                # Draft cache also needs sync
                # Reset draft cache to match target
                # Optimized: just reset cur_pos and copy?
                # Copying is safer.

            # Sync draft
            draft.cur_pos = self.cur_pos
            # Copy just the accepted region?
            # Or assume draft cache is dirty and needs refresh?
            # Draft cache has garbage at the end now.
            # Copy valid region from Target to Draft
            # Valid region is up to self.cur_pos
            # Only need to copy the *new* valid tokens (n_accepted + 1)
            # from target to draft.

            # Optimized copy:
            # target cache at [start_pos_before : self.cur_pos]
            start_copy = start_pos_before
            end_copy = self.cur_pos
            if end_copy > start_copy:
                 draft.kv_cache[:, :n_d, :, start_copy:end_copy, :, :] = self.kv_cache[:, :n_d, :, start_copy:end_copy, :, :]

            if generated_count >= max_new_tokens:
                break

        return generated
