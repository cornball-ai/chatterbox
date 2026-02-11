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

| Backend | Cold Start | Warm Start | Audio | Real-time Factor | VRAM |
|---------|-----------|------------|-------|------------------|------|
| Container (Python) | 1.4s | 1.3s | 3.2s | **2.5x** | 3,205 MB |
| Native R (Rtorch) | 4.3s | 3.6s | 4.8s | **1.3x** | 3,114 MB |
| Native R + compile | 3.9s | 4.0s | 4.5s | **1.1x** | 4,575 MB |

### Speedup vs Native R

| Backend | Speedup |
|---------|---------|
| Compiled | 0.89x (slower) |
| Container | 2.8x |

The compiled backend uses `Rtorch::compile()` (torchlang) to fuse individual
sub-module forward passes into compiled IR graphs. In practice this does not
help because the bottleneck is per-token autoregressive loop overhead, not
individual module dispatch. The compiled MLP/Mish/GELU operations are already
dominated by libtorch CUDA kernel execution time.

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
| Native R | 3,114 MB | ~3,200 MB |
| Compiled | 4,575 MB | ~4,600 MB |
| Container | ~3,200 MB | ~3,200 MB |

The CUDA caching allocator may hold additional reserved memory between
generations. Use `Rtorch::cuda_empty_cache()` to release it back to the
driver. Use `Rtorch::cuda_memory_stats()` for per-process allocated/reserved
breakdown.

## When to Use Each

### Use Native R When:

- No Docker available or desired
- Long-running R sessions (model stays cached)
- Full control over inference parameters
- Custom fine-tuning or LoRA experimentation

Native R with Rtorch now runs faster than real-time (1.3x RT factor),
making it practical for interactive use.

### Use Container When:

- Speed is critical (~2.8x faster than native)
- Production deployments
- GPU resource management via gpu.ctl

## Reproducing

Run the benchmark script from the chatterbox package directory:

```bash
r scripts/benchmark_gpu.R
```

This compares Rtorch CUDA inference (plain and compiled) against the
Python container (if running). For VRAM profiling:

```bash
r scripts/vram_profile.R
```

## Future Improvements

Potential optimizations not yet implemented:

1. **float16 inference**: Would halve memory bandwidth requirements
2. **Loop-level compilation**: Compiling the full autoregressive loop
   (not just individual sub-modules) could eliminate the remaining R
   overhead per token
3. **Flash Attention 2**: Requires custom CUDA kernels beyond SDPA

The R-to-C++ boundary overhead (~1.4 us per `.Call()`) is the floor for
Rtorch. The remaining ~2.8x gap vs Python is primarily in per-token loop
overhead (R control flow between autoregressive steps) and Python's
optimized kernel dispatch.
