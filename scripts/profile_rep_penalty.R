#!/usr/bin/env r
# Profile repetition penalty implementations

library(torch)

device <- "cuda"
vocab_size <- 8194
repetition_penalty <- 1.2

# Simulate different sequence lengths
for (seq_len in c(10, 50, 100, 200)) {
    logits <- torch::torch_randn(1, vocab_size, device = device)
    generated_ids <- torch::torch_randint(1, vocab_size, c(1, seq_len), device = device)

    # Current implementation (R for loop)
    torch::cuda_synchronize()
    t1 <- Sys.time()
    for (iter in 1:20) {
        logits_copy <- logits$clone()
        for (token_id in as.integer(generated_ids$cpu())) {
            logits_copy[1, token_id] <- logits_copy[1, token_id] / repetition_penalty
        }
    }
    torch::cuda_synchronize()
    loop_time <- as.numeric(Sys.time() - t1) * 1000 / 20

    # Vectorized implementation
    torch::cuda_synchronize()
    t1 <- Sys.time()
    for (iter in 1:20) {
        logits_copy <- logits$clone()
        # Get unique token IDs
        unique_ids <- unique(as.integer(generated_ids$cpu()))
        # Create penalty tensor
        penalties <- torch::torch_ones(vocab_size, device = device)
        for (id in unique_ids) {
            penalties[id] <- repetition_penalty
        }
        logits_copy <- logits_copy / penalties$unsqueeze(1)
    }
    torch::cuda_synchronize()
    vec_time <- as.numeric(Sys.time() - t1) * 1000 / 20

    # Fully vectorized with scatter
    torch::cuda_synchronize()
    t1 <- Sys.time()
    for (iter in 1:20) {
        logits_copy <- logits$clone()
        # Get unique IDs on CPU
        unique_ids <- unique(as.integer(generated_ids$cpu()))
        n_unique <- length(unique_ids)
        # Index directly
        logits_copy[1, unique_ids] <- logits_copy[1, unique_ids] / repetition_penalty
    }
    torch::cuda_synchronize()
    scatter_time <- as.numeric(Sys.time() - t1) * 1000 / 20

    cat(sprintf("seq_len=%3d: loop=%.1fms, vec=%.1fms, scatter=%.1fms\n",
                seq_len, loop_time, vec_time, scatter_time))
}
