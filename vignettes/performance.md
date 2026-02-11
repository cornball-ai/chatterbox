---
title: Performance Comparison: Native R vs Container
---

# Performance Comparison

The chatterbox package provides native R inference for the Chatterbox TTS model
using [Rtorch](https://github.com/cornball-ai/Rtorch) for direct libtorch
bindings. This vignette compares performance between the native R implementation
and the Python container backend.

## Test Configuration

- **GPU**: RTX 5060 Ti (16GB VRAM, Blackwell architecture)
- **Model**: Chatterbox TTS (798M parameters)
- **Precision**: float32 (all backends, including container)
- **Test text**: "The quick brown fox jumps over the lazy dog." (~10 words)
- **Tensor library**: Rtorch 0.1.0 (raw `.Call()` to libtorch, no Rcpp/lantern)

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

Each tensor operation in R crosses the R/C++ boundary via `.Call()`:

```r
# Each of these is a separate .Call() into libtorch
x <- x$add(y)
x <- x$mul(z)
x <- Rtorch::nnf_gelu(x)
```

Rtorch eliminates the intermediate layers that the `torch` R package uses
(R7 dispatch, lantern shim, Rcpp glue), bringing per-operation overhead
from ~10 us down to ~1.4 us. But Python's pybind11 path is still faster
at ~0.5 us per operation. With ~100-200 tokens and ~30 transformer layers,
this adds up.

### 2. Autoregressive Generation

Token generation is inherently sequential -- each token depends on the
previous. This means:

- ~100-200 sequential forward passes through 30 transformer layers
- No opportunity for batch parallelism
- Per-operation overhead multiplied by every token

### 3. JIT Tracing

The traced backend compiles transformer layers and CFM estimator into C++
graphs, eliminating per-operation R overhead within those modules. The
generation loop itself still runs in R.

## Native Backend Details

### Pure R (`backend = "r"`)

All inference runs in R. Each tensor operation individually crosses the
R/C++ boundary via Rtorch's `.Call()` interface. ~500ms per token.

### C++ T3 Decode (`backend = "cpp"`)

The T3 autoregressive decode loop runs in C++ via libtorch, eliminating R
overhead for the token generation phase. S3Gen vocoder still runs in R.
~225ms per token.

Requires libtorch headers at install time (auto-detected by the configure
script).

### R Traced (`traced = TRUE`)

Uses `jit_trace()` to compile both T3 transformer layers and CFM estimator
into C++ graphs.

```r
result <- generate(model, text, voice, traced = TRUE)
```

**Cold start** (~84s): Traces 30 T3 transformer layers + KV projectors, plus
the CFM estimator at fixed max length. This is a one-time cost per session.

**Warm start** (~13s): Runs the traced graphs directly. ~130ms per token.

**Limitations:**
- T3 cache limited to 350 tokens (including conditioning ~50-100 tokens)
- CFM max sequence length 1024 (longer sequences fall back to non-traced)
- Uses more VRAM (~4.2GB vs ~3.1GB) due to cached traced modules

## Optimizations Applied

### SDPA (Scaled Dot-Product Attention)

```r
Rtorch::torch_scaled_dot_product_attention(q, k, v)
```

Calls libtorch's fused SDPA kernel which selects the best available
implementation (Flash Attention, efficient attention, or math fallback).

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

### T3/VE Offloading During S3Gen

After T3 token generation completes, the T3 model and voice encoder are
moved to CPU and the CUDA cache is cleared before S3Gen inference. This
reduces peak VRAM during the vocoder pass:

```r
model$t3$to(device = "cpu")
model$voice_encoder$to(device = "cpu")
gc(); gc()
Rtorch::cuda_empty_cache()
# ... S3Gen inference ...
```

## Memory Usage

| Backend | Model VRAM | Peak During Generation |
|---------|-----------|----------------------|
| Pure R / C++ | 3,112 MB | 3,133 MB |
| Traced | 4,211 MB | 4,238 MB |
| Container | ~3,000 MB | ~3,200 MB |

The CUDA caching allocator may hold additional reserved memory between
generations. Use `Rtorch::cuda_empty_cache()` to release it back to the
driver. Use `Rtorch::cuda_memory_stats()` for per-process allocated/reserved
breakdown.

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

## Reproducing

Run the benchmark script from the chatterbox package directory:

```bash
r scripts/benchmark_gpu.R
```

This compares Rtorch CUDA inference against the Python container (if
running). For VRAM profiling:

```bash
r scripts/vram_profile.R
```

## Future Improvements

Potential optimizations not yet implemented:

1. **float16 inference**: Would halve memory bandwidth requirements
2. **Fused kernels**: Custom CUDA kernels for attention + FFN blocks
3. **Flash Attention 2**: Requires custom CUDA kernels beyond SDPA

The R-to-C++ boundary overhead (~1.4 us per `.Call()`) is the floor for
Rtorch. JIT tracing helps by batching operations into a single C++ graph
execution. The remaining ~3x gap vs Python is in per-call overhead and
Python's optimized kernel dispatch.
