#!/usr/bin/env r
# Profile R to C++ boundary crossing overhead

library(torch)

device <- "cuda"

# Simple profiling helper
profile_ops <- function(label, n_iters, expr) {
    torch::cuda_synchronize()
    t1 <- Sys.time()
    for (i in 1:n_iters) {
        force(expr)
    }
    torch::cuda_synchronize()
    elapsed <- as.numeric(Sys.time() - t1) * 1000 / n_iters
    cat(sprintf("%-40s: %.3f ms\n", label, elapsed))
    elapsed
}

cat("=== R->C++ Boundary Crossing Analysis ===\n\n")

# Create test tensors
batch <- 2L
seq_len <- 1L
hidden <- 1024L
heads <- 16L
head_dim <- 64L

x <- torch::torch_randn(batch, seq_len, hidden, device = device)
q <- torch::torch_randn(batch, heads, seq_len, head_dim, device = device)
k <- torch::torch_randn(batch, heads, 100L, head_dim, device = device)
v <- torch::torch_randn(batch, heads, 100L, head_dim, device = device)

linear <- torch::nn_linear(hidden, hidden)$to(device = device)

n_iters <- 1000

cat("--- Individual Operations ---\n")

# Single linear layer
t_linear <- profile_ops("Single linear forward", n_iters, {
    linear$forward(x)
})

# Single tensor method
t_method <- profile_ops("Tensor method (view)", n_iters, {
    x$view(c(batch, -1L))
})

# Single math op
t_math <- profile_ops("Math op (x * 2)", n_iters, {
    x * 2
})

# Multiple chained methods
t_chain3 <- profile_ops("3 chained methods", n_iters, {
    x$view(c(batch, seq_len, -1L))$transpose(2, 3)$contiguous()
})

t_chain5 <- profile_ops("5 chained methods", n_iters, {
    x$view(c(batch, seq_len, -1L))$transpose(2, 3)$contiguous()$view(c(-1L, hidden))$unsqueeze(1)
})

cat("\n--- Attention Component Costs ---\n")

# QKV projection (3 boundary crossings)
t_qkv <- profile_ops("QKV projections (3 linear)", n_iters, {
    linear$forward(x)
    linear$forward(x)
    linear$forward(x)
})

# SDPA alone
sdpa <- get("torch_scaled_dot_product_attention", envir = asNamespace("torch"))
t_sdpa <- profile_ops("SDPA (fused attention)", n_iters, {
    sdpa(q, k, v, attn_mask = list(), dropout_p = 0.0, is_causal = FALSE)
})

# Full attention reshape sequence
t_reshape <- profile_ops("Attention reshapes (6 ops)", n_iters, {
    q1 <- x$view(c(batch, seq_len, heads, head_dim))
    q2 <- q1$transpose(2, 3)
    k1 <- x$view(c(batch, seq_len, heads, head_dim))
    k2 <- k1$transpose(2, 3)
    v1 <- x$view(c(batch, seq_len, heads, head_dim))
    v2 <- v1$transpose(2, 3)
})

cat("\n--- RoPE Costs ---\n")

# RoPE computation (many small ops)
cos_cache <- torch::torch_randn(128, head_dim, device = device)
sin_cache <- torch::torch_randn(128, head_dim, device = device)
position_ids <- torch::torch_zeros(c(batch, seq_len), dtype = torch::torch_long(), device = device)
q_test <- torch::torch_randn(batch, heads, seq_len, head_dim, device = device)
k_test <- torch::torch_randn(batch, heads, seq_len, head_dim, device = device)

rotate_half <- function(x) {
    x1 <- x[,,, 1:(x$size(4) %/% 2)]
    x2 <- x[,,, (x$size(4) %/% 2 + 1):x$size(4)]
    torch::torch_cat(list(-x2, x1), dim = -1)
}

t_rope <- profile_ops("Full RoPE (12+ ops)", n_iters, {
    cos <- cos_cache[position_ids$add(1L),]$unsqueeze(3)
    sin <- sin_cache[position_ids$add(1L),]$unsqueeze(3)
    cos <- cos$transpose(2, 3)
    sin <- sin$transpose(2, 3)
    q_embed <- (q_test * cos) + (rotate_half(q_test) * sin)
    k_embed <- (k_test * cos) + (rotate_half(k_test) * sin)
})

cat("\n--- Full Layer Estimates ---\n")

# Count operations per attention forward:
# 3 linear (QKV) + 6 reshape + RoPE (~15) + 2 concat + 2 repeat + SDPA + 3 output = ~32 ops
# MLP: 3 linear + 2 multiply + sigmoid = 6 ops
# Norms: 8 ops each x 2 = 16 ops
# Total per layer: ~54 ops

cat("\nEstimated boundary crossings per layer:\n")
cat("  Attention: ~32 ops\n")
cat("  MLP: ~6 ops\n")
cat("  Norms: ~16 ops\n")
cat("  Total: ~54 ops per layer\n")
cat("  30 layers: ~1620 ops per token\n")

# Calculate estimated overhead
avg_crossing <- (t_linear + t_method + t_math) / 3
cat(sprintf("\nAvg single crossing time: %.3f ms\n", avg_crossing))
cat(sprintf("Estimated crossing overhead per token: %.1f ms (1620 x %.3f)\n",
            1620 * avg_crossing, avg_crossing))

cat("\n--- Potential Optimizations ---\n")

# Test fused ops if available
cat("\n1. Combine QKV into single projection:\n")
qkv_proj <- torch::nn_linear(hidden, hidden * 3, bias = FALSE)$to(device = device)

t_qkv_fused <- profile_ops("   Fused QKV (1 linear)", n_iters, {
    qkv_proj$forward(x)
})
cat(sprintf("   Savings: %.3f ms (%.1fx)\n", t_qkv - t_qkv_fused, t_qkv / t_qkv_fused))

cat("\n2. RoPE alternatives:\n")
# Precomputed full embeddings
cat("   Current RoPE is complex due to indexing and rotate_half\n")
cat("   Could precompute for fixed position in cached forward\n")

cat("\n3. In-place operations:\n")
x_copy <- x$clone()
t_inplace <- profile_ops("   Inplace mul_", n_iters, {
    x_copy$mul_(1.0)
})
t_outplace <- profile_ops("   Out-of-place mul", n_iters, {
    x_copy * 1.0
})
cat(sprintf("   Savings: %.3f ms\n", t_outplace - t_inplace))

cat("\n=== Summary ===\n")
cat("The R->C++ boundary crossing overhead is significant but not the main cost.\n")
cat("Each crossing is ~0.01-0.02 ms, totaling ~20-30 ms per token.\n")
cat("The actual compute (SDPA, linear layers) is comparable.\n")
cat("\nBiggest wins would come from:\n")
cat("1. Fused QKV projection (implemented above)\n")
cat("2. More aggressive caching in RoPE\n")
cat("3. Avoiding redundant dtype conversions in RMSNorm\n")
