#!/usr/bin/env r
# Profile time spent in each transformer component

library(torch)

rhydrogen::load_all("/home/troy/chatterbox")

model <- chatterbox("cuda")
model <- load_chatterbox(model)

llama <- model$t3$tfmr
device <- "cuda"

# Create dummy inputs
batch_size <- 2
seq_len <- 1  # Single token (cached forward)
hidden_dim <- 1024

hidden_states <- torch::torch_randn(batch_size, seq_len, hidden_dim, device = device)
position_ids <- torch::torch_zeros(c(batch_size, seq_len), dtype = torch::torch_long(), device = device)

# Get rope cache
rope <- compute_rope_frequencies(64, 1000, device = device)

# Create fake past_kv for one layer
past_len <- 100
fake_k <- torch::torch_randn(batch_size, 16, past_len, 64, device = device)
fake_v <- torch::torch_randn(batch_size, 16, past_len, 64, device = device)
fake_past <- list(k = fake_k, v = fake_v)

# Get first layer
layer <- llama$layers[[1]]

# Warm up
for (i in 1:5) {
    out <- layer$forward(hidden_states, position_ids, rope$cos, rope$sin, NULL, fake_past)
}

cat("=== Single Layer Breakdown (cached forward) ===\n\n")

n_iters <- 50

# Time full layer
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    out <- layer$forward(hidden_states, position_ids, rope$cos, rope$sin, NULL, fake_past)
}
torch::cuda_synchronize()
layer_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Full layer: %.2f ms\n", layer_time))

# Time input norm
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    normed <- layer$input_layernorm$forward(hidden_states)
}
torch::cuda_synchronize()
norm1_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("  Input norm: %.2f ms\n", norm1_time))

# Time attention (self_attn)
attn <- layer$self_attn
normed <- layer$input_layernorm$forward(hidden_states)

torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    attn_out <- attn$forward(normed, position_ids, rope$cos, rope$sin, NULL, fake_past)
}
torch::cuda_synchronize()
attn_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("  Attention: %.2f ms\n", attn_time))

# Break down attention
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    q <- attn$q_proj$forward(normed)
    k <- attn$k_proj$forward(normed)
    v <- attn$v_proj$forward(normed)
}
torch::cuda_synchronize()
qkv_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("    QKV projection: %.2f ms\n", qkv_time))

# Time post-attention norm
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    normed2 <- layer$post_attention_layernorm$forward(hidden_states)
}
torch::cuda_synchronize()
norm2_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("  Post-attn norm: %.2f ms\n", norm2_time))

# Time MLP
mlp <- layer$mlp
normed2 <- layer$post_attention_layernorm$forward(hidden_states)

torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    mlp_out <- mlp$forward(normed2)
}
torch::cuda_synchronize()
mlp_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("  MLP: %.2f ms\n", mlp_time))

cat(sprintf("\nEstimated 30-layer forward: %.1f ms\n", layer_time * 30))
cat(sprintf("Actual observed: ~160 ms\n"))
