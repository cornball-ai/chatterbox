#!/usr/bin/env r
# Profile softmax and min-p filtering

library(torch)

device <- "cuda"
vocab_size <- 8194
min_p <- 0.05
top_p <- 0.95
temperature <- 0.8

logits <- torch::torch_randn(1, vocab_size, device = device)

# Current: two softmax calls
torch::cuda_synchronize()
t1 <- Sys.time()
for (iter in 1:100) {
    logits_scaled <- logits / temperature

    # First softmax for min-p
    probs <- torch::nnf_softmax(logits_scaled, dim = -1)
    max_prob <- probs$max()
    min_threshold <- min_p * max_prob
    logits_filtered <- logits_scaled$clone()
    logits_filtered[probs < min_threshold] <- -65504.0

    # Second softmax after min-p
    probs_filtered <- torch::nnf_softmax(logits_filtered, dim = -1)

    # Top-p
    sorted_result <- torch::torch_sort(probs_filtered, descending = TRUE)
}
torch::cuda_synchronize()
two_softmax <- as.numeric(Sys.time() - t1) * 1000 / 100
cat(sprintf("Two softmax approach: %.2f ms\n", two_softmax))

# Alternative: single softmax, mask in prob space
torch::cuda_synchronize()
t1 <- Sys.time()
for (iter in 1:100) {
    logits_scaled <- logits / temperature

    # Single softmax
    probs <- torch::nnf_softmax(logits_scaled, dim = -1)

    # Min-p in prob space
    max_prob <- probs$max()
    min_threshold <- min_p * max_prob
    probs_filtered <- probs$clone()
    probs_filtered[probs < min_threshold] <- 0

    # Renormalize
    probs_filtered <- probs_filtered / probs_filtered$sum()

    # Top-p
    sorted_result <- torch::torch_sort(probs_filtered, descending = TRUE)
}
torch::cuda_synchronize()
one_softmax <- as.numeric(Sys.time() - t1) * 1000 / 100
cat(sprintf("One softmax approach: %.2f ms\n", one_softmax))

cat(sprintf("Savings: %.2f ms (%.1f%%)\n", two_softmax - one_softmax,
            (two_softmax - one_softmax) / two_softmax * 100))
