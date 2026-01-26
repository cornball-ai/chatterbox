#!/usr/bin/env r
# Simple T3 profiling - focus on transformer forward pass

library(torch)

rhydrogen::load_all("/home/troy/chatterbox")

model <- chatterbox("cuda")
model <- load_chatterbox(model)

t3 <- model$t3
device <- "cuda"

# Create dummy input for transformer
batch_size <- 2  # CFG doubled
seq_len <- 50
hidden_dim <- t3$config$n_channels  # 1024

dummy_embeds <- torch::torch_randn(batch_size, seq_len, hidden_dim, device = device)

cat("=== Transformer Forward Timing ===\n\n")

# Initial forward (builds KV cache)
torch::cuda_synchronize()
t1 <- Sys.time()
output <- t3$tfmr$forward(inputs_embeds = dummy_embeds, use_cache = TRUE)
torch::cuda_synchronize()
init_time <- as.numeric(Sys.time() - t1) * 1000
cat(sprintf("Initial forward (%d tokens): %.1f ms\n", seq_len, init_time))

# Cached forward (single token)
past_kv <- output$past_key_values
single_token_emb <- torch::torch_randn(batch_size, 1, hidden_dim, device = device)

# Warm up cached forward
for (i in 1:5) {
    output2 <- t3$tfmr$forward(inputs_embeds = single_token_emb,
                                past_key_values = past_kv,
                                use_cache = TRUE)
}

# Time cached forward
torch::cuda_synchronize()
t1 <- Sys.time()
n_iters <- 20
for (i in 1:n_iters) {
    output2 <- t3$tfmr$forward(inputs_embeds = single_token_emb,
                                past_key_values = past_kv,
                                use_cache = TRUE)
}
torch::cuda_synchronize()
cached_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Cached forward (1 token): %.1f ms\n", cached_time))

# Time head projection
hidden <- output2$last_hidden_state
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    logits <- t3$speech_head$forward(hidden)
}
torch::cuda_synchronize()
head_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Head projection: %.1f ms\n", head_time))

# Time softmax
logits_squeezed <- logits$squeeze(2)
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    probs <- torch::nnf_softmax(logits_squeezed / 0.8, dim = -1)
}
torch::cuda_synchronize()
softmax_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Softmax: %.1f ms\n", softmax_time))

# Time sort (for top-p)
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    sorted <- torch::torch_sort(probs, descending = TRUE)
}
torch::cuda_synchronize()
sort_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Sort: %.1f ms\n", sort_time))

# Time cumsum
sorted_probs <- sorted[[1]]
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    cumsum <- torch::torch_cumsum(sorted_probs, dim = -1)
}
torch::cuda_synchronize()
cumsum_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Cumsum: %.1f ms\n", cumsum_time))

# Time multinomial sampling
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:n_iters) {
    sample <- torch::torch_multinomial(probs, num_samples = 1)
}
torch::cuda_synchronize()
sample_time <- as.numeric(Sys.time() - t1) * 1000 / n_iters
cat(sprintf("Multinomial: %.1f ms\n", sample_time))

cat("\n=== Summary ===\n")
total_components <- cached_time + head_time + softmax_time + sort_time + cumsum_time + sample_time
cat(sprintf("Sum of components: %.1f ms\n", total_components))
cat(sprintf("Actual per-token: ~232 ms (from full inference)\n"))
cat(sprintf("Unexplained overhead: ~%.1f ms\n", 232 - total_components))
