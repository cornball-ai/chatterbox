# JIT Trace Optimization for R torch

## Problem

R torch has significant per-operation overhead from R-to-C++ boundary crossing:

| Operation | R (µs) | Python (µs) | Ratio |
|-----------|--------|-------------|-------|
| tensor.size() | 7.7 | 0.1 | 77x |
| tensor.mul() | 18.8 | 3.5 | 5.4x |
| nn.Linear | 106.5 | 74.4 | 1.4x |

This results in ~10x slower inference for transformers.

## Solution: jit_trace

`torch::jit_trace()` compiles R torch code to a C++ graph, eliminating per-operation R overhead.

### Results

| Component | Normal R | Traced | Speedup |
|-----------|----------|--------|---------|
| Single layer (no cache) | 2.43ms | 0.30ms | **8x** |
| Full transformer (no cache) | 1105ms | 96.5ms | **11.5x** |
| Single layer (with pre-alloc cache) | 2.64ms | 0.31ms | **8.5x** |

### The KV Cache Challenge

Standard KV caching uses dynamic tensor shapes (growing each token), which breaks `jit_trace`.

**Solution: Pre-allocated cache with masking**

1. Pre-allocate KV cache to maximum length (e.g., 300 tokens)
2. Use attention mask to indicate valid positions
3. Trace with fixed shapes
4. Update cache values and mask each step

```r
# Pre-allocate
k_cache <- torch::torch_zeros(batch, heads, max_len, head_dim, device = device)
v_cache <- torch::torch_zeros(batch, heads, max_len, head_dim, device = device)
valid_mask <- torch::torch_zeros(batch, 1, 1, max_len, dtype = torch::torch_bool(), device = device)

# Each step:
# 1. Compute new K, V
# 2. Write to cache at current position: k_cache[,, pos, ] <- new_k
# 3. Update mask: valid_mask[,,, 1:pos] <- TRUE
# 4. Run traced forward
```

## Implementation Plan

### Phase 1: Trace First Token (Quick Win)
- Trace full transformer without KV cache
- Use for first forward pass (conditioning tokens)
- Benefit: ~10x faster first token

### Phase 2: Pre-allocated Cache (Full Solution)
- Implement pre-allocated cache with masking
- Trace full transformer with cache inputs
- Update cache in R between traced calls
- Expected: ~8x faster per-token, matching Python

### Architecture Changes Needed

1. **llama.R**: Add `llama_attention_preallocated` variant
2. **t3.R**: Add `t3_inference_traced` that uses pre-allocated cache
3. **tts.R**: Option to use traced inference

### Memory Impact

Pre-allocated cache for 300 tokens:
- Per layer: 2 × batch × heads × 300 × head_dim × 4 bytes
- 30 layers: 2 × 2 × 16 × 300 × 64 × 4 × 30 = 147MB
- Acceptable tradeoff for 8x speedup

## Code Example

```r
# Traceable layer with pre-allocated cache
traceable_layer <- torch::nn_module(
    forward = function(hidden_states, position_ids, cos, sin,
                       k_cache, v_cache, valid_mask) {
        # ... QKV projection and RoPE ...

        # SDPA with mask
        attn_mask <- torch::torch_where(
            valid_mask,
            torch::torch_zeros(1, device = device),
            torch::torch_tensor(-1e9, device = device)
        )
        attn_output <- sdpa(q, k_cache, v_cache, attn_mask = attn_mask)

        # ... rest of layer ...
    }
)

# Create and trace
wrapped <- traceable_layer(layer)
traced <- torch::jit_trace(wrapped, ...)

# Use in generation loop
for (step in 1:max_tokens) {
    # Run traced forward
    hidden <- traced(hidden, pos, cos, sin, k_cache, v_cache, valid_mask)

    # Update cache (R operations, outside trace)
    k_cache[,, step + cond_len, ] <- new_k
    v_cache[,, step + cond_len, ] <- new_v
    valid_mask[,,, step + cond_len] <- TRUE
}
```

## Limitations

1. **torch.compile not available**: R torch doesn't have Python's `torch.compile()`
2. **CUDA graphs not exposed**: Could provide additional speedup
3. **Cache updates in R**: Still some R overhead between traced calls

## References

- R torch jit_trace: https://torch.mlverse.org/docs/reference/jit_trace.html
- PyTorch TorchScript: https://pytorch.org/docs/stable/jit.html
