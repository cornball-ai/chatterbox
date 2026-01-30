#!/usr/bin/env r
# Detailed profiling of R->C++ boundary crossings

library(torch)

rhydrogen::load_all("/home/troy/chatterbox")

cat("Loading model...\n")
model <- chatterbox("cuda")
model <- load_chatterbox(model)

llama <- model$t3$tfmr
device <- "cuda"

# Create inputs for cached forward (single token)
batch_size <- 2L
seq_len <- 1L
hidden_dim <- 1024L

hidden_states <- torch::torch_randn(batch_size, seq_len, hidden_dim, device = device)
position_ids <- torch::torch_zeros(c(batch_size, seq_len), dtype = torch::torch_long(), device = device)

# Get rope cache
rope <- compute_rope_frequencies(64, 1000, device = device)

# Create fake past_kv for one layer
past_len <- 100L
fake_k <- torch::torch_randn(batch_size, 16, past_len, 64, device = device)
fake_v <- torch::torch_randn(batch_size, 16, past_len, 64, device = device)
fake_past <- list(k = fake_k, v = fake_v)

layer <- llama$layers[[1]]
attn <- layer$self_attn
mlp <- layer$mlp

# High-precision timing
precise_time <- function(label, n_iters, fn) {
    # Warm up
    for (i in 1:10) fn()

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

cat("\n=== Fine-Grained Attention Breakdown ===\n")
cat("(cached forward with seq_len=1, past_len=100)\n\n")

# Individual projections
t_q <- precise_time("Q projection only", n_iters, function() {
    attn$q_proj$forward(hidden_states)
})

t_k <- precise_time("K projection only", n_iters, function() {
    attn$k_proj$forward(hidden_states)
})

t_v <- precise_time("V projection only", n_iters, function() {
    attn$v_proj$forward(hidden_states)
})

t_o <- precise_time("O projection only", n_iters, function() {
    attn$o_proj$forward(hidden_states)
})

# Reshapes
q_proj <- attn$q_proj$forward(hidden_states)
t_reshape <- precise_time("Q reshape (view+transpose)", n_iters, function() {
    q_proj$view(c(batch_size, seq_len, 16L, 64L))$transpose(2, 3)
})

# RoPE (the expensive part?)
q_shaped <- q_proj$view(c(batch_size, seq_len, 16L, 64L))$transpose(2, 3)
k_shaped <- torch::torch_randn(batch_size, 16, seq_len, 64, device = device)

t_rope <- precise_time("RoPE application (Q and K)", n_iters, function() {
    apply_rotary_pos_emb(q_shaped, k_shaped, rope$cos, rope$sin, position_ids)
})

# KV cache concat
new_k <- torch::torch_randn(batch_size, 16, seq_len, 64, device = device)
new_v <- torch::torch_randn(batch_size, 16, seq_len, 64, device = device)
t_concat <- precise_time("KV cache concat (2 cats)", n_iters, function() {
    torch::torch_cat(list(fake_past$k, new_k), dim = 3)
    torch::torch_cat(list(fake_past$v, new_v), dim = 3)
})

# SDPA
q_for_sdpa <- torch::torch_randn(batch_size, 16, seq_len, 64, device = device)
k_for_sdpa <- torch::torch_randn(batch_size, 16, 101, 64, device = device)
v_for_sdpa <- torch::torch_randn(batch_size, 16, 101, 64, device = device)
sdpa <- get("torch_scaled_dot_product_attention", envir = asNamespace("torch"))

t_sdpa <- precise_time("SDPA kernel", n_iters, function() {
    sdpa(q_for_sdpa, k_for_sdpa, v_for_sdpa, attn_mask = list(), dropout_p = 0.0, is_causal = FALSE)
})

# Output reshape
attn_out <- torch::torch_randn(batch_size, 16, seq_len, 64, device = device)
t_out_reshape <- precise_time("Output reshape (transpose+view)", n_iters, function() {
    attn_out$transpose(2, 3)$contiguous()$view(c(batch_size, seq_len, hidden_dim))
})

cat("\n--- Attention Sum ---\n")
sum_attn <- t_q + t_k + t_v + t_reshape * 3 + t_rope + t_concat + t_sdpa + t_out_reshape + t_o
cat(sprintf("Sum of parts: %.4f ms\n", sum_attn))

# Full attention
t_full_attn <- precise_time("Full attention forward", n_iters, function() {
    attn$forward(hidden_states, position_ids, rope$cos, rope$sin, NULL, fake_past)
})

cat(sprintf("Measured full: %.4f ms\n", t_full_attn))
cat(sprintf("Overhead from aggregation: %.4f ms (%.1f%%)\n",
            t_full_attn - sum_attn, (t_full_attn - sum_attn) / t_full_attn * 100))

cat("\n=== MLP Breakdown ===\n")

mlp_input <- torch::torch_randn(batch_size, seq_len, hidden_dim, device = device)

t_gate <- precise_time("Gate projection", n_iters, function() {
    mlp$gate_proj$forward(mlp_input)
})

t_up <- precise_time("Up projection", n_iters, function() {
    mlp$up_proj$forward(mlp_input)
})

gate_out <- mlp$gate_proj$forward(mlp_input)
t_silu <- precise_time("SiLU activation", n_iters, function() {
    gate_out * torch::torch_sigmoid(gate_out)
})

t_down <- precise_time("Down projection", n_iters, function() {
    mlp$down_proj$forward(mlp_input)
})

cat("\n--- MLP Sum ---\n")
sum_mlp <- t_gate + t_up + t_silu + t_down
cat(sprintf("Sum of parts: %.4f ms\n", sum_mlp))

t_full_mlp <- precise_time("Full MLP forward", n_iters, function() {
    mlp$forward(mlp_input)
})
cat(sprintf("Measured full: %.4f ms\n", t_full_mlp))

cat("\n=== RMSNorm Breakdown ===\n")

norm <- layer$input_layernorm
norm_input <- torch::torch_randn(batch_size, seq_len, hidden_dim, device = device)

t_dtype <- precise_time("To float32", n_iters, function() {
    norm_input$to(dtype = torch::torch_float32())
})

x_f32 <- norm_input$to(dtype = torch::torch_float32())
t_pow <- precise_time("pow(2)", n_iters, function() {
    x_f32$pow(2)
})

t_mean <- precise_time("mean(dim=-1, keepdim=TRUE)", n_iters, function() {
    x_f32$pow(2)$mean(dim = -1, keepdim = TRUE)
})

variance <- x_f32$pow(2)$mean(dim = -1, keepdim = TRUE)
t_rsqrt <- precise_time("rsqrt(variance + eps)", n_iters, function() {
    torch::torch_rsqrt(variance + 1e-5)
})

t_full_norm <- precise_time("Full RMSNorm", n_iters, function() {
    norm$forward(norm_input)
})

cat("\n=== Full Layer Analysis ===\n")

t_full_layer <- precise_time("Full decoder layer", n_iters, function() {
    layer$forward(hidden_states, position_ids, rope$cos, rope$sin, NULL, fake_past)
})

cat(sprintf("\n30-layer estimate: %.2f ms per token\n", t_full_layer * 30))

cat("\n=== R->C++ Crossing Count ===\n")
cat("\nPer attention forward:\n")
cat("  4 linear projections (Q,K,V,O)\n")
cat("  6 reshape ops (3 view + 3 transpose)\n")
cat("  ~15 RoPE ops (indexing, transpose, multiply, cat)\n")
cat("  2 KV cache concat\n")
cat("  1 SDPA\n")
cat("  3 output reshape (transpose, contiguous, view)\n")
cat("  = ~31 crossings\n")

cat("\nPer MLP forward:\n")
cat("  3 linear projections\n")
cat("  2 multiplies (SwiGLU)\n")
cat("  1 sigmoid\n")
cat("  = 6 crossings\n")

cat("\nPer RMSNorm:\n")
cat("  2 dtype conversions\n")
cat("  pow, mean, rsqrt, multiply, multiply\n")
cat("  = ~7 crossings\n")

cat("\nPer decoder layer:\n")
cat("  2 norms = 14\n")
cat("  1 attention = 31\n")
cat("  1 MLP = 6\n")
cat("  2 residual adds = 2\n")
cat("  = 53 crossings per layer\n")
cat("  = 1590 crossings for 30 layers\n")

cat("\n=== Optimization Ideas ===\n")

cat("\n1. FUSED QKV: Combine Q,K,V into single linear\n")
fused_qkv <- torch::nn_linear(hidden_dim, hidden_dim * 3, bias = FALSE)$to(device = device)
t_fused_qkv <- precise_time("   Fused QKV projection", n_iters, function() {
    fused_qkv$forward(hidden_states)
})
cat(sprintf("   Current (3 separate): %.4f ms\n", t_q + t_k + t_v))
cat(sprintf("   Fused (1 combined): %.4f ms\n", t_fused_qkv))
cat(sprintf("   Savings: %.4f ms per layer, %.2f ms per token\n",
            (t_q + t_k + t_v) - t_fused_qkv,
            ((t_q + t_k + t_v) - t_fused_qkv) * 30))

cat("\n2. PRECOMPUTE RoPE for position 0 (cached forward always uses pos 0)\n")
cat("   Current RoPE: %.4f ms\n", t_rope)
cat("   Could cache cos_0, sin_0 tensors for position 0\n")

cat("\n3. SKIP dtype conversion in RMSNorm if already float32\n")
cat(sprintf("   Dtype conversion overhead: %.4f ms per norm\n", t_dtype * 2))
cat(sprintf("   30 layers x 2 norms x 2 conversions = %.2f ms potential savings\n", t_dtype * 4 * 30))
