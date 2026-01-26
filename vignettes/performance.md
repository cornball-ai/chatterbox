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
- **Test text**: ~20 words generating ~6-8 seconds of audio (~200 tokens)

## Results Summary

| Implementation | Precision | Generation Time | Audio Length | Real-time Factor |
|----------------|-----------|-----------------|--------------|------------------|
| Container (Python) | float16 | 2.2s | 6s | **2.7x** |
| Native R (traced) | float32 | 15.6s | 5.4s | **0.35x** |
| Native R (normal) | float32 | 34s | 5.2s | **0.15x** |

**Traced inference is 2.2x faster** than normal R inference.
The container is still approximately **8x faster** than traced native R.

## Why the Difference?

### 1. Precision (float16 vs float32)

The container uses float16 (half precision), which:

- Halves memory bandwidth requirements
- Enables tensor core acceleration on modern GPUs
- Reduces VRAM usage (3.4GB vs 3.3GB - similar due to other overhead)

R torch currently lacks easy float16 inference support for custom models.

### 2. R-to-C++ Boundary Overhead

Each tensor operation in R crosses the R/C++ boundary:

```r
# Each of these is a separate C++ call
x <- x$add(y)
x <- x$mul(z)
x <- nnf_gelu(x)
```

Python's tighter C++ integration has lower per-operation overhead. With ~200 tokens
and ~30 transformer layers, this adds up to significant overhead.

### 3. Autoregressive Generation

Token generation is inherently sequential - each token depends on the previous.
This means:

- ~200 sequential forward passes through 30 transformer layers
- No opportunity for batch parallelism
- R overhead multiplied by every token

## Optimizations Applied

The native R implementation includes several optimizations that improved
performance from 232ms/token to 135ms/token (42% faster):

### SDPA (Scaled Dot-Product Attention)

```r
# Using fused attention kernel (2.7x faster in isolation)
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

### Single Softmax for Min-p Filtering

```r
# Before: Two softmax calls
probs1 <- nnf_softmax(logits, dim = -1)
# ... filter in logit space ...
probs2 <- nnf_softmax(filtered_logits, dim = -1)

# After: One softmax, filter in probability space
probs <- nnf_softmax(logits, dim = -1)
probs[probs < threshold] <- 0
probs <- probs / probs$sum()  # Simple renormalization
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

## When to Use Each

### Use Native R When:

- No Docker available or desired
- Long-running R sessions (model stays cached)
- Custom fine-tuning or LoRA experimentation
- Full control over inference parameters
- Offline operation required

### Use Container When:

- Speed is critical (10x faster)
- Short-lived scripts (avoid 17s model load each time)
- Production deployments
- GPU resource management via gpu.ctl

## Memory Usage

| Implementation | Peak VRAM | Steady-state VRAM |
|----------------|-----------|-------------------|
| Container | ~3.7GB | 3.4GB |
| Native R (optimized) | ~3.3GB | 3.3GB |

The CPU-first loading optimization eliminated the 6GB peak that occurred when
weights were temporarily held in both the state dict and model.

## JIT Trace Optimization (Jan 2026)

The `traced = TRUE` parameter enables JIT-traced inference, which compiles R torch
code to a C++ graph and eliminates per-operation R overhead.

### How it works

1. **Pre-allocated KV cache**: Fixed-size cache (350 tokens) with attention mask
2. **Traced layers**: Each transformer layer + KV projector is traced once
3. **Generation loop**: Update cache values and mask in R, run traced forward

```r
# Enable traced inference
result <- tts(model, text, voice, traced = TRUE)
```

### Limitations

- Cache limited to 350 tokens (including conditioning)
- First call compiles traced modules (one-time overhead)
- Longer sequences may be truncated

### Benchmark

Same hardware, same text (~5s audio):

| Mode | Time | Real-time Factor | Speedup |
|------|------|------------------|---------|
| Normal | 34.0s | 0.15x | baseline |
| Traced | 15.6s | 0.35x | **2.2x** |

## Future Improvements

Potential optimizations not yet implemented:

1. **float16 inference**: Would require careful dtype management throughout
2. **torch.compile()**: Not available in R torch
3. **Flash Attention 2**: Requires custom CUDA kernels
4. **Speculative decoding**: Would add complexity

The R-to-C++ boundary overhead is fundamental and cannot be eliminated without
changes to R torch itself. JIT tracing helps by batching operations into a single
C++ graph execution.
