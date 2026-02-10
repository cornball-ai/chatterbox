---
title: Performance Comparison: Native R vs Container
---

# Performance Comparison

The chatterbox package provides native R inference for the Chatterbox TTS model.
This vignette compares performance between the native R implementation and the
Python container backend.

## Test Configuration

- **GPU**: RTX 5060 Ti (16GB VRAM, Blackwell architecture)
- **Model**: Chatterbox TTS (798M parameters)
- **Precision**: float32 (all backends, including container)
- **Test text**: "The quick brown fox jumps over the lazy dog." (~10 words)

## Results Summary

| Backend | Cold Start | Warm Start | Audio | Real-time Factor |
|---------|-----------|------------|-------|------------------|
| Container (Python) | 1.1s | 1.3s | 3.1s | **2.5x** |
| Native R (traced) | 83.8s | 12.6s | 4.0s | **0.32x** |
| Native R (C++ T3) | 26.6s | 26.7s | 4.2s | **0.16x** |
| Native R (pure R) | 50.5s | 53.2s | 5.3s | **0.10x** |

Cold start includes one-time JIT compilation overhead for traced mode.

### Speedups (warm start, relative to pure R)

| Backend | Speedup |
|---------|---------|
| C++ T3 decode | 2.0x |
| R traced | 4.2x |
| Container | 41.7x |

## Why the Difference?

### 1. R-to-C++ Boundary Overhead

Each tensor operation in R crosses the R/C++ boundary:

```r
# Each of these is a separate C++ call
x <- x$add(y)
x <- x$mul(z)
x <- nnf_gelu(x)
```

Python's tighter C++ integration has lower per-operation overhead. With ~100-200
tokens and ~30 transformer layers, this adds up significantly.

### 2. Autoregressive Generation

Token generation is inherently sequential - each token depends on the previous.
This means:

- ~100-200 sequential forward passes through 30 transformer layers
- No opportunity for batch parallelism
- R overhead multiplied by every token

### 3. JIT Tracing

The traced backend compiles transformer layers and CFM estimator into C++ graphs,
eliminating per-operation R overhead within those modules. The generation loop
itself still runs in R.

## Native Backend Details

### Pure R (`backend = "r"`)

All inference runs in R. Each tensor operation individually crosses the R/C++
boundary. ~500ms per token.

### C++ T3 Decode (`backend = "cpp"`)

The T3 autoregressive decode loop runs in C++ via libtorch, eliminating R overhead
for the token generation phase. S3Gen vocoder still runs in R. ~225ms per token.

Requires libtorch headers at install time (auto-detected by the configure script).

### R Traced (`traced = TRUE`)

Uses `torch::jit_trace()` to compile both T3 transformer layers and CFM estimator
into C++ graphs.

```r
result <- generate(model, text, voice, traced = TRUE)
```

**Cold start** (~84s): Traces 30 T3 transformer layers + KV projectors, plus the
CFM estimator at fixed max length. This is a one-time cost per session.

**Warm start** (~13s): Runs the traced graphs directly. ~130ms per token.

**Limitations:**
- T3 cache limited to 350 tokens (including conditioning ~50-100 tokens)
- CFM max sequence length 1024 (longer sequences fall back to non-traced)
- Uses more VRAM (~4.2GB vs ~3.1GB) due to cached traced modules

## Optimizations Applied

The native R implementation includes several optimizations:

### SDPA (Scaled Dot-Product Attention)

```r
# Using fused attention kernel
torch::torch_scaled_dot_product_attention(q, k, v)
```

Note: This function exists in R torch but is not exported. We access it via
`get()` from the torch namespace.

### Vectorized Repetition Penalty

```r
# Before: O(n) loop
for (token_id in generated_ids) {
    logits[1, token_id] <- logits[1, token_id] / penalty
}

# After: O(1) vectorized
unique_ids <- unique(as.integer(generated_ids$cpu()))
logits[1, unique_ids] <- logits[1, unique_ids] / penalty
```

### CPU-First Weight Loading

```r
# Load to CPU first, then move to GPU
# This halves peak VRAM (avoids weights in both dict and model)
weights <- read_safetensors(path, device = "cpu")
model <- load_weights(model, weights)
rm(weights); gc()
model$to(device = "cuda")
```

## Memory Usage

| Backend | Model VRAM | Peak During Generation |
|---------|-----------|----------------------|
| Pure R / C++ | 3,112 MB | 3,133 MB |
| Traced | 4,211 MB | 4,238 MB |
| Container | ~3,000 MB | ~3,200 MB |

The CUDA caching allocator may hold additional reserved memory between generations.
Use `cuda_empty_cache()` to release it back to the driver.

## When to Use Each

### Use Container When:

- Speed is critical (~10x faster than best native)
- Production deployments
- GPU resource management via gpu.ctl

### Use Native R (traced) When:

- No Docker available or desired
- Long-running R sessions (one-time 84s compilation, then fast)
- Need full control over inference parameters

### Use Native R (C++ T3) When:

- Traced mode uses too much VRAM
- Generating very long sequences (>350 tokens)

### Use Native R (pure R) When:

- libtorch headers not available at install time
- Debugging or development
- Custom fine-tuning or LoRA experimentation

## Future Improvements

Potential optimizations not yet implemented:

1. **float16 inference**: Would halve memory bandwidth requirements
2. **torch.compile()**: Not available in R torch
3. **Flash Attention 2**: Requires custom CUDA kernels

The R-to-C++ boundary overhead is fundamental and cannot be eliminated without
changes to R torch itself. JIT tracing helps by batching operations into a single
C++ graph execution.
