# Einsum Transformer

A **NumPy-based platform for evaluating Transformer-style models** (Llama, Mistral, Qwen, etc.) without PyTorch or CUDA. It maps **EINSUM equations directly to execution**—no custom optimized kernels—so you can compare architectures and run training/inference on **multithreaded CPU with SIMD** (via NumPy’s BLAS/Accelerate backends). All experimentation was done on **Apple Silicon (M5)**; the code is portable and can be run on Linux, Windows, or other Macs.

## Why this repo?

Typical stacks (PyTorch, JAX, CUDA) hide the “heavy” layers behind optimized kernels. This project **simplifies that pipeline**: the same EINSUM you write on paper is what runs. There are no custom CUDA/MLIR kernels—just **NumPy** (vectorized ops and SIMD), **OpenMP-style multithreading** (via BLAS/`OMP_NUM_THREADS`), and **Ray** for data-parallel fwd/bwd. That makes it easier to:

- **Evaluate** different Transformer variants (GQA, MQA, MoE, RoPE, LoRA, quantization) in one place
- **Debug and teach** the exact tensor ops without framework abstractions
- **Run on CPU-only** hardware (e.g. Mac Silicon, Xeon, EPYC) with good utilization

## Requirements

The runtime has two main dependencies:

| Component | Role |
|-----------|------|
| **NumPy** | Vectorized operations and **SIMD** through BLAS (Accelerate on macOS, OpenBLAS/MKL elsewhere). All EINSUM/matmul-heavy work goes through NumPy. |
| **OpenMP** | **Multithreading** for BLAS/NumPy. Set `OMP_NUM_THREADS` (and `VECLIB_MAXIMUM_THREADS` on macOS) via `--num-threads` before import. |
| **Ray** (optional) | **Data-parallel** forward/backward over batch shards (`--ray-workers N`). Use `pip install ray` or `uv add einsum[ray]`. |

Install: `uv sync` (or `pip install -e .`); for Ray: `uv sync --extra ray` or `pip install ray`.

## Intent & Use Case

This repository serves as a transparent "glass box" for LLM execution flow. It allows for:
- **Parameter Sensitivity Analysis**: Tweak RoPE base frequencies, context lengths, and model dimensions.
- **Micro-Architectural Bottleneck Identification**: Granular profiling helps identify exactly where CPU cycles are spent (e.g., Weight Loading vs. MatMuls).
- **Architectural Backpropagation Research**: Manually derived `backward` passes for every component allow for direct research into gradient flow and optimization strategies.
- **Component Sub-Phasing**: Visualize the transition between Prefill, Decode, and Training stages.

## Design Philosophy

### The "Einstein" Approach (`einsum`)
We use `np.einsum` as the definitive notation for all tensor operations. 
- **Readability**: Operations like `btd,dh->bth` map directly to paper equations.
- **Accuracy**: Eliminates ambiguity associated with nested `transpose` and `matmul` calls.
- **Backprop Integrity**: The same `einsum` notation used for the forward pass is derived for the backward pass, ensuring mathematical symmetry.
- **HW Optimization**: Utilizes NumPy's underlying BLAS/MKL/Accelerate backends for efficient SIMD execution on corresponding hardware.

### Memory Management & Bfloat16
To run models like Llama-3.2 on standard hardware:
- **Lazy Loading**: Weights are indexed across multiple shards and loaded on-demand, minimizing peak memory usage.
- **Bfloat16 Support**: Native compatibility with modern high-precision weight formats using `ml-dtypes`.

### Mixed-precision quantization (WxAy)

Config `quantization` enables mixed-precision in the forward path. Format: **W{weight_bits}A{activation_bits}** (e.g. `W8A16`, `W4A4`). The notation is **source (original) → target (quantized)**.

- **Weights (W4, W8)**: Symmetric int4/int8 quantize→dequant to float32 before einsum. Weights are effectively stored/transferred in lower precision; compute remains float32.
- **Activations (A4, A8, A16)**: A16 = fp16 cast; A8/A4 = int8/int4 quantize→dequant at block boundaries (after embed, before attention, before FFN). Activations are reduced at those boundaries; downstream ops run in float32.
- **Config**: `"quantization": "W8A16"` in JSON; or `optmizations.enable_quantization(cfg, "W4A4")`.

#### Impact on accuracy

- **W8A16**: Small impact; W8 adds rounding in weights, A16 matches common training/inference practice. Often used as a safe default.
- **W4A16 / W8A8**: Larger impact; W4 and A8 increase rounding and dynamic-range loss. Quality depends on model and calibration; useful for size/speed vs quality tradeoffs.
- **W4A4 / W8A4**: Highest compression, highest risk of degradation; A4 is very aggressive. Mainly for memory-bound or exploratory setups.

In this codebase, quantization is **simulated** (quantize→dequant to float32). It reflects precision loss, not yet real int4/int8 matmuls, so you can study accuracy/behavior before moving to lower-level backends.

#### Impact on performance

- **This implementation**: Quantize/dequant adds extra passes (round, scale, clip). Throughput can be slightly worse than `quantization: "none"` when compute is in float32. The gain is in **memory**, not in faster matmuls here.
- **With real low-precision kernels**: INT8/FP16 matmuls (e.g. on GPU/TensorCores) typically improve throughput and reduce memory bandwidth. The Mem save numbers below approximate what such backends can achieve.

#### Impact on memory capacity

The per-layer breakdown and **Mem save** show the **theoretical** reduction vs a float32 baseline (4 bytes/element). Assumed: float32=4 B, fp16=2 B, int8=1 B, int4=0.5 B.

**Mem save is total across all layers** (not per layer): Embedding is once; Attn, FFN, and RMSNorm are summed over all `n_layers`. So Total can reach several GB for large models (e.g. ~9 GB for Llama 3.2 3B with W8A16: ~12 GB in float32 → ~3 GB in W8 + activation savings).

**Example 1: `fast_lora` (d_model=512, 8 layers) with `W8A16` — total ~105 MB:**

```
  Operation          Weights            Activations        Mem save
  ------------------ ------------------ ------------------ --------------
  Embedding          float32→int8       float32→fp16       6.500 MB
  RMSNorm (×2)       float32→int8       float32            0.023 MB
  Attn (Q,K,V,O)     float32→int8       float32→fp16       22.000 MB
  FFN (W1,W2,W3)     float32→int8       float32→fp16       76.000 MB
  ---------------------------------------------------------------------------
  Total (all layers)                                             104.523 MB
```

**Example 2: `llama` (Llama 3.2 3B, d_model=3072, 28 layers) with `W8A16` — total ~9.2 GB:**

```
  Operation          Weights            Activations        Mem save
  ------------------ ------------------ ------------------ --------------
  Embedding          bfloat16→int8      float32→fp16       1128.000 MB
  RMSNorm (×2)       bfloat16→int8      float32            0.492 MB
  Attn (Q,K,V,O)     bfloat16→int8      float32→fp16       2037.000 MB
  FFN (W1,W2,W3)     bfloat16→int8      float32→fp16       6069.000 MB
  ---------------------------------------------------------------------------
  Total (all layers)                                             9234.492 MB
```

- **Embedding**: E (V×d) + embedding output (B×T×d). One block for the whole model.
- **Attn / FFN / RMSNorm**: Per-layer weight and activation (B×T×d at block input) savings, **summed over all layers**; hence large totals for 3B-scale models.

With `quantization: "none"`, Mem save is `—`. Run e.g. `python run_model.py --config configs/fast_lora.json` or `--config configs/llama.json` (set `"quantization": "W8A16"`) to see the live table.

## Framework Architecture

### Execution Flow
```
    Token Input
         |
         v
   Embedding Lookup
         |
         v
   Layer Loop 0...N
         |
         v
   +------------------ Transformer Layer --------------------+
   |     RMS Norm                                            |
   |        |                                                |
   |        v                                                |
   |  Einsum Attention  <----------------->  KV Cache        |
   |        |                                                |
   |        v                                                |
   |     RMS Norm                                            |
   |        |                                                |
   |        v                                                |
   |  Einsum FeedForward / SwiGLU                            |
   |        |                                                |
   |        v                                                |
   |  Residual Add                                           |
   +---------------------------------------------------------+
         |
         v
   Final RMS Norm
         |
         v
   Output Head
         |
         v
   Top-K Sampler
         |
         v
   De-tokenizer
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Lazy Shard Loading** | Automatic discovery and on-demand loading of sharded `.safetensors`. |
| **Iterative Decode Loop** | Full generation flow from prompt (Prefill) to iterative token sampling. |
| **Silent Decode Mode** | Streamlined per-token logging to reduce terminal noise during generation. |
| **Aggregate Summaries** | Performance reporting across the entire generation budget, not just per-step. |
| **RoPE Instrumentation** | Visual feedback on Base frequency, Head dimension, and Position indices. |
| **Top-K Sampling** | Controllable generation with configurable temperature and sampling budget. |
| **Manual Backprop** | Full component-wise `backward()` pass implemented using pure `einsum` notation. |
| **Einsum Optimizer** | In-memory weight updates via a transparent SGD-based `EinsumOptimizer`. |

## Architecture & Feature Matrix

Characterization of architectural options and which models they map to:

| Feature | Status | Example models |
|---------|--------|----------------|
| **GQA** (Grouped Query Attention) | Implemented | Llama 2/3, Mistral, Qwen, Phi-2/3, Gemma |
| **MQA** (Multi-Query Attention) | Implemented | Phi-1, MPT |
| **Speculative decoding** | Hooked | Llama, Gemma, Mistral (inference-only; draft config in `optmizations`) |
| **MoE FFN** (Mixture of Experts) | Implemented | Mixtral, Llama 3.1 8B/405B, Qwen2-MoE, DeepSeek-MoE |
| **Mixed-precision quantization (WxAy)** | Implemented | Llama, Phi, Gemma, Mistral (config `quantization`: W8A16, W8A8, W4A4, W8A4, etc.) |
| **Prefill/decode split** | Hooked | All (config `prefill_only` for hardware split; prefill/decode phases implemented) |
| **RoPE** (Rotary Positional Embeddings) | Implemented | Llama, Mistral, Qwen, Phi, Gemma |
| **RMSNorm** | Implemented | Llama, Mistral, Qwen, Phi, Gemma |
| **SwiGLU** | Implemented | Llama, Mistral, Qwen, Phi, Gemma |
| **KV cache** | Implemented | All (Llama, Phi, Gemma, Mistral, Qwen) |
| **Weight tying** (embed + LM head) | Implemented | Llama, Phi, GPT-2, Gemma |
| **Flash tiling** (tiled matmul) | Already integrated | All (via NumPy/BLAS in `einsum`) |

Configs: `use_gqa`, `use_mqa`, `use_moe`, `quantization`; `optmizations.enable_*` for speculative, quantization, prefill-only.

## Telemetry & Profiling Breakdown

The framework breaks down performance into logical architectural phases. Below is a guide to interpreting the timing breakdown table:

| Phase | Description | Key Weights Involved |
|-------|-------------|----------------------|
| **QKV Proj** | Projecting the input hidden state into Query, Key, and Value vectors. | `q_proj`, `k_proj`, `v_proj` |
| **RoPE** | Applying Rotary Positional Embeddings to the Q and K tensors. | _Positional logic_ |
| **Score/Sftmx** | Computing attention scores (dot product) and applying the Softmax function. | _Intermediate tensors_ |
| **Out Agg/Proj** | Aggregating Values and projecting back to the hidden dimension. | `o_proj` |
| **Ld weights** | Time to access weights (inside Attention / FFN). ~0 when in-memory (random init or cached); non-zero on first load from safetensors. | _Various shards_ |
| **FFN Compute** | The SwiGLU feed-forward pass (W1/W2/W3 projections). | `gate_proj`, `up_proj`, `down_proj` |
| **Prefill Stage** | Time to process the entire input prompt (KV cache generation). | _Complete pass_ |
| **Decode Stage** | Average/Peak time to generate a single new token using the KV cache. | _Iterative loop_ |

## Setup & Usage

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/shanksjay/einsum-transformer
cd einsum-transformer

# Install dependencies using uv
uv sync
```

### 2. Quick validation (fast config)

Use `configs/fast_lora.json` to verify your environment quickly (small model, no external weights, ~4–6s inference, ~1 min training for 10 steps):

```bash
# Inference: 10 tokens, 8 layers (config default), random init
python run_model.py --config configs/fast_lora.json

# Training: 10 steps, 4 layers, fixed batch, seed 42 (loss should decrease)
python train_model.py --lr 0.05 --steps 10 --config configs/fast_lora.json --max-layers 4 --fix-batch --seed 42
```

Prepend `uv run` if using uv (e.g. `uv run python run_model.py ...`).

### 3. Configuration (`configs/llama.json`)
You can control the architecture and generation parameters directly via JSON:
```json
{
  "max_new_tokens": 10,
  "weight_dtype": "bfloat16",
  "seq_len": 128,
  "hf_repo_id": "meta-llama/Llama-3.2-3B-Instruct",
  "temperature": 1.0,
  "top_k_sample": 50
}
```

### 4. Running Inference
```bash
uv run python run_model.py --config configs/llama.json
```

### 5. Running Training Experiments
```bash
# Run a synthetic training loop to verify gradient flow
uv run python train_model.py --steps 5 --lr 1e-4
```

### 6. CPU utilization and multithreading

Training is CPU-bound (NumPy/BLAS). To improve core usage:

- **`--num-threads N`** (recommended first): Sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and on macOS `VECLIB_MAXIMUM_THREADS` before any NumPy import. This increases parallelism in matmuls and often gives the largest gain.
  ```bash
  uv run python train_model.py --num-threads 8 --steps 5 --config configs/fast_lora.json
  ```
  Use a value close to your P-cores or P+E cores (e.g. 8–10 on Apple M5).

- **`--ray-workers N`** (optional): Data-parallel forward/backward over batch shards. Requires `ray` (`pip install ray` or `uv add einsum[ray]`). Best with in-memory or LoRA models; each worker holds a model copy.
  ```bash
  uv run python train_model.py --ray-workers 4 --num-threads 4 --config configs/fast_lora.json --steps 5
  ```

#### Benchmark: `--ray-workers` 1 vs 4

Same workload: `fast_lora`, 4 layers, 5 steps, `--fix-batch --seed 42 --num-threads 4`.

| `--ray-workers` | Total time | Mean step time | Notes |
|-----------------|------------|----------------|-------|
| **1** (single process) | **39.3 s** | **7.86 s/step** | Baseline |
| **4** (data-parallel)  | **~21–24 s** | **~4.2–4.8 s/step** | ~1.6–1.9× faster; requires `pip install ray` |

With 4 workers, the batch is sharded across 4 processes; each runs forward+backward on a 1/4 batch, then the main process averages gradients and runs the optimizer. Speedup is typically 1.5–2× depending on cores, serialization cost, and batch size. Use `--ray-workers 4 --num-threads 4` so each worker uses 4 BLAS threads.

To reproduce:
```bash
# Single process (baseline)
python train_model.py --config configs/fast_lora.json --steps 5 --max-layers 4 --seed 42 --fix-batch --num-threads 4 --ray-workers 1

# 4 Ray workers (requires: pip install ray)
python train_model.py --config configs/fast_lora.json --steps 5 --max-layers 4 --seed 42 --fix-batch --num-threads 4 --ray-workers 4
```

## Training & Backpropagation

The platform provides a fully derivative-enabled sandbox:
- **`backward()`**: Every component (Attention, RMSNorm, SwiGLU) has a manually derived `einsum` backward pass.
- **GQA Aware**: Gradients are correctly accumulated across groups for Grouped Query Attention.
- **In-Memory Cache**: The `LazyWeightManager` includes a caching layer that enables weight updates to models originally loaded from read-only `safetensors`.

## LoRA Support & Efficient Fine-Tuning

The platform supports **Low-Rank Adaptation (LoRA)** for efficient fine-tuning. This drastically reduces memory usage by freezing the base model and training only small low-rank adapter matrices.

### Key Benefits
- **Minimal Parameters**: Train <0.1% of parameters (2.3M vs 3.2B for Llama 3B).
- **Gradient Freezing**: The `backward` pass automatically skips gradient computation for frozen weights, reducing gradient memory usage from ~24GB to ~50MB.
- **Merge Capability**: Adapters can be merged back into the base weights for efficient deployment.

### Comparison (Llama-3.2 3B)

| Metric | Full Fine-Tuning | LoRA Training (r=8) |
|--------|------------------|---------------------|
| **Trainable Params** | ~3.21 Billion | ~2.29 Million |
| **Percentage** | 100% | 0.07% |
| **Gradient Memory** | ~24.5 GB | ~50 MB |
| **Optimizer State** | ~49.0 GB | ~18 MB |

### Usage

1. **Configure LoRA**: Add LoRA fields to your config (`configs/my_lora.json`):
```json
{
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "lora_target_modules": ["q_proj", "v_proj"],
  "lora_merge_after_training": true
}
```

2. **Run Training**:
```bash
uv run python train_model.py --config configs/my_lora.json --steps 10
```

3. **Merging**: If `lora_merge_after_training` is true, adapters are merged into base weights at the end of training. You can also call `model.merge_lora_weights()` manually.

##  Example Output & Performance Analysis

### Inference Mode Output (e.g. fast_lora, 10 tokens)
```
==================================================
TIMING BREAKDOWN (seconds)
--------------------------------------------------
Phase                | p50        | p99       
----------------------------------------------
Embedding            | 0.0003     | 0.0003    
Layer Total          | 0.7005     | 0.7183    
  Norm               | 0.0003     | 0.0004    
  Weight Load        | 0.0000     | 0.0000    
  Attention          | 0.1433     | 0.1463    
    Ld weights       | 0.0000     | 0.0000    
    QKV Proj         | 0.0821     | 0.0838    
    RoPE             | 0.0011     | 0.0012    
    Score/Sftmx      | 0.0092     | 0.0096    
    Out Agg/Proj     | 0.0512     | 0.0550    
  FeedForward        | 0.5555     | 0.5715    
    Ld weights       | 0.0000     | 0.0000    
    Compute          | 0.5555     | 0.5715    
Output Head          | 0.3376     | 0.3376    
----------------------------------------------
[NOTE] Prefill components sum: 5.9156s, Total: 5.5781s, Diff: -0.3376s (overhead/KV cache)
Prefill Stage        | 5.5781     | 5.5781    
----------------------------------------------
KV Cache Size        | 16.00      MB
==================================================
```

**Key Observations:**
- **Embedding** and **Output Head** are timed separately; the [NOTE] explains when components sum ≠ Prefill total (output head / KV work).
- **FFN dominates** layer compute: FeedForward (0.56s) vs Attention (0.14s).
- **Attention**: QKV Proj + Out Agg/Proj account for most of attention time; RoPE and Score/Sftmx are small.
- **KV Cache Size**: `n_layers × 2 × batch × n_heads × seq_len × d_head` (× 4 bytes); reported after prefill populates the cache.

### Training Mode Output (e.g. fast_lora, 4 layers, 10 steps)

Training runs with `verbose=False` per step; the **TRAINING SUMMARY** and the **forward/backward timing tables** are printed once at the end (p50/p99 over all steps).

```
==================================================
TRAINING SUMMARY
==================================================
Total Steps:         10
Total Time:          48.50s
Step time (fwd+bwd+opt):  mean=4.850s  p50=4.599s  p99=5.882s
Loss:                start=8.3277  end=8.3434
Optimizer mem/step:  272.06 MB (opt update only)
Gradient Memory:     136.03 MB

==================================================
TIMING BREAKDOWN (seconds)
--------------------------------------------------
Phase                | p50        | p99       
----------------------------------------------
Embedding            | 0.0001     | 0.0001    
Layer Total          | 0.6769     | 0.7109    
  Norm               | 0.0005     | 0.0005    
  Weight Load        | 0.0000     | 0.0000    
  Attention          | 0.1400     | 0.1473    
  FeedForward        | 0.5499     | 0.5708    
Output Head          | 0.3155     | 0.3155    
----------------------------------------------
[NOTE] Prefill components sum: 3.04s, Total: 2.73s, Diff: -0.32s (overhead/KV cache)
Prefill Stage        | 2.7273     | 2.7273    
==================================================

==================================================
BACKWARD PASS TIMING BREAKDOWN (seconds)
--------------------------------------------------
Phase                | p50        | p99       
----------------------------------------------
Backward Pass        | 3.1462     | 3.1462    
  BW Layers          | 0.6705     | 0.7975    
  BW Head            | 0.3345     | 0.3345    
  BW Embed           | 0.0003     | 0.0003    
  BW Attention       | 0.1747     | 0.3012    
  BW FeedForward     | 0.4946     | 0.4951    
  BW Norms           | 0.0005     | 0.0008    
----------------------------------------------
Gradient Memory      | 136.03     MB
==================================================
```

**Key Observations:**
- **TRAINING SUMMARY**: Total Time = sum of step times; Step time is fwd+bwd+opt; Loss start/end shows if training is learning (use `--fix-batch --seed 42` to see loss decrease).
- **Forward table**: Same phases as inference; p50/p99 over all steps.
- **Backward table**: BW Head, BW Embed, BW Attention, BW FeedForward, BW Norms; Gradient Memory matches summary.
- **Optimizer mem/step**: Memory for the optimizer update only, not fwd/bwd activations.

## Parameter Experimentation Guide

Understanding how architectural parameters impact performance:

### 1. Sequence Length Impact
```bash
# Short sequences (fast prefill, less memory)
python run_model.py --config configs/llama.json  # seq_len: 128

# Edit config to test longer sequences
# seq_len: 512 → expect 4x prefill time (quadratic attention)
# seq_len: 1024 → expect 16x prefill time
```
**Expected Impact:**
- Attention scores: O(seq_len²) - dominates for long sequences
- KV cache memory: Linear growth with seq_len
- FFN compute: Linear with seq_len

### 2. Layer Count Impact
```bash
# Quick experiments with limited layers
python train_model.py --max-layers 2  # Fast iteration

# Full model training
python train_model.py  # All 28 layers
```
**Expected Impact:**
- Total time scales linearly with layers
- Gradient memory scales linearly (one set of gradients per layer)
- Useful for isolating per-layer costs

### 3. Batch Size Impact
```bash
# Edit config: "batch_size": 1 (default)
# Edit config: "batch_size": 4
# Edit config: "batch_size": 8
```
**Expected Impact:**
- FFN compute: Near-linear scaling (good SIMD utilization)
- Attention: Scales with batch size but maintains O(T²) per sequence
- Memory: Linear growth in activations and gradients

### 4. Learning Rate & Optimizer Analysis
```bash
# Conservative update
python train_model.py --lr 1e-5 --steps 10

# Aggressive update
python train_model.py --lr 1e-3 --steps 10
```
**Watch for:**
- Loss progression: Should decrease smoothly with proper LR
- Optimizer timing: Should be consistent across LR values
- Memory stays constant (memory is architecture-dependent, not LR-dependent)

### 5. Profiling Bottlenecks

Compare these timing ratios to identify optimization opportunities:

| Metric | What to Look For | Action |
|--------|------------------|--------|
| `Weight Load / Total` | >20% suggests I/O bound | Consider weight preloading or caching |
| `FFN / Attention` ratio | >3:1 typical for dense models | FFN is optimization target |
| `Backward / Forward` ratio | Should be 0.5-1.5x forward | If >2x, check gradient computation |
| `p99 / p50` ratio | >2x suggests variance | Investigate inconsistent layer costs |

### 6. Memory Profiling
```bash
# Monitor gradient memory growth
python train_model.py --max-layers 1  # Baseline
python train_model.py --max-layers 2  # 2x gradient memory
python train_model.py --max-layers 4  # 4x gradient memory
```
**Expected patterns:**
- Gradient memory should scale linearly with layers
- Optimizer memory/step = 2× gradient memory (weights + grads)
- Forward activations are cleared after backward pass

---
