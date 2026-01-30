#!/usr/bin/env r
# Profile KV cache operations - the actual bottleneck

library(torch)

device <- "cuda"

cat("=== KV Cache Bottleneck Analysis ===\n\n")

batch_size <- 2L
heads <- 16L
head_dim <- 64L
seq_len <- 1L  # New token

# High-precision timing
precise_time <- function(label, n_iters, fn) {
    for (i in 1:10) fn()  # Warmup
    torch::cuda_synchronize()
    t1 <- Sys.time()
    for (i in 1:n_iters) {
        fn()
    }
    torch::cuda_synchronize()
    elapsed <- as.numeric(Sys.time() - t1) * 1000 / n_iters
    cat(sprintf("%-50s: %.4f ms\n", label, elapsed))
    elapsed
}

n_iters <- 500

cat("--- KV Cache Concat Costs by Past Length ---\n\n")

for (past_len in c(10, 50, 100, 200, 500)) {
    past_k <- torch::torch_randn(batch_size, heads, past_len, head_dim, device = device)
    past_v <- torch::torch_randn(batch_size, heads, past_len, head_dim, device = device)
    new_k <- torch::torch_randn(batch_size, heads, seq_len, head_dim, device = device)
    new_v <- torch::torch_randn(batch_size, heads, seq_len, head_dim, device = device)

    t <- precise_time(sprintf("KV concat (past_len=%d)", past_len), n_iters, function() {
        torch::torch_cat(list(past_k, new_k), dim = 3)
        torch::torch_cat(list(past_v, new_v), dim = 3)
    })
}

cat("\n--- Alternative: Pre-allocated KV Cache (Python style) ---\n\n")

max_len <- 512L
past_len <- 100L

# Pre-allocate full cache
kv_cache_k <- torch::torch_zeros(batch_size, heads, max_len, head_dim, device = device)
kv_cache_v <- torch::torch_zeros(batch_size, heads, max_len, head_dim, device = device)

# Fill initial past
kv_cache_k[,, 1:past_len,] <- torch::torch_randn(batch_size, heads, past_len, head_dim, device = device)
kv_cache_v[,, 1:past_len,] <- torch::torch_randn(batch_size, heads, past_len, head_dim, device = device)

new_k <- torch::torch_randn(batch_size, heads, seq_len, head_dim, device = device)
new_v <- torch::torch_randn(batch_size, heads, seq_len, head_dim, device = device)

# Write new position in-place
t_inplace <- precise_time("In-place KV write (pre-allocated)", n_iters, function() {
    pos <- past_len + 1L
    kv_cache_k[,, pos,] <- new_k[,, 1,]
    kv_cache_v[,, pos,] <- new_v[,, 1,]
})

# Compare with concat
past_k_dynamic <- torch::torch_randn(batch_size, heads, past_len, head_dim, device = device)
past_v_dynamic <- torch::torch_randn(batch_size, heads, past_len, head_dim, device = device)

t_concat <- precise_time("Dynamic concat (current method)", n_iters, function() {
    torch::torch_cat(list(past_k_dynamic, new_k), dim = 3)
    torch::torch_cat(list(past_v_dynamic, new_v), dim = 3)
})

cat(sprintf("\nSpeedup from pre-allocation: %.1fx\n", t_concat / t_inplace))
cat(sprintf("Savings per layer: %.4f ms\n", t_concat - t_inplace))
cat(sprintf("Savings for 30 layers: %.2f ms per token\n", (t_concat - t_inplace) * 30))

cat("\n--- Alternative: slice_scatter ---\n\n")

# Try slice_scatter if available
tryCatch({
    kv_cache_k2 <- torch::torch_zeros(batch_size, heads, max_len, head_dim, device = device)
    t_scatter <- precise_time("slice_scatter approach", n_iters, function() {
        kv_cache_k2$slice_scatter(new_k, dim = 3, start = past_len, end = past_len + 1)
    })
}, error = function(e) {
    cat("slice_scatter not available\n")
})

cat("\n--- RoPE Optimization ---\n\n")

# Current RoPE with dynamic position lookup
cos_cache <- torch::torch_randn(1000, head_dim, device = device)
sin_cache <- torch::torch_randn(1000, head_dim, device = device)
position_ids <- torch::torch_tensor(matrix(past_len, nrow = batch_size, ncol = 1),
                                     dtype = torch::torch_long(), device = device)
q <- torch::torch_randn(batch_size, heads, seq_len, head_dim, device = device)
k <- torch::torch_randn(batch_size, heads, seq_len, head_dim, device = device)

rotate_half <- function(x) {
    x1 <- x[,,, 1:(x$size(4) %/% 2)]
    x2 <- x[,,, (x$size(4) %/% 2 + 1):x$size(4)]
    torch::torch_cat(list(-x2, x1), dim = -1)
}

t_rope_full <- precise_time("Full RoPE (dynamic position)", n_iters, function() {
    cos <- cos_cache[position_ids$add(1L),]$unsqueeze(3)
    sin <- sin_cache[position_ids$add(1L),]$unsqueeze(3)
    cos <- cos$transpose(2, 3)
    sin <- sin$transpose(2, 3)
    q_embed <- (q * cos) + (rotate_half(q) * sin)
    k_embed <- (k * cos) + (rotate_half(k) * sin)
})

# Pre-extract single position
cos_pos <- cos_cache[past_len + 1,]$unsqueeze(1)$unsqueeze(1)$unsqueeze(1)
sin_pos <- sin_cache[past_len + 1,]$unsqueeze(1)$unsqueeze(1)$unsqueeze(1)

t_rope_cached <- precise_time("RoPE with pre-extracted position", n_iters, function() {
    q_embed <- (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_embed <- (k * cos_pos) + (rotate_half(k) * sin_pos)
})

cat(sprintf("\nRoPE speedup from position caching: %.1fx\n", t_rope_full / t_rope_cached))
cat(sprintf("Savings for 30 layers: %.2f ms\n", (t_rope_full - t_rope_cached) * 30))

cat("\n=== Summary of Bottlenecks ===\n\n")
cat("Current per-token costs (30 layers, past_len=100):\n")
cat(sprintf("  KV cache concat: %.2f ms (BIGGEST BOTTLENECK)\n", t_concat * 30))
cat(sprintf("  RoPE computation: %.2f ms\n", t_rope_full * 30))
cat("\nWith optimizations:\n")
cat(sprintf("  Pre-allocated KV cache: saves %.2f ms (%.0f%% of concat time)\n",
            (t_concat - t_inplace) * 30, (t_concat - t_inplace) / t_concat * 100))
cat(sprintf("  Cached RoPE position: saves %.2f ms\n", (t_rope_full - t_rope_cached) * 30))
cat(sprintf("  Total potential savings: %.2f ms per token\n",
            (t_concat - t_inplace) * 30 + (t_rope_full - t_rope_cached) * 30))
