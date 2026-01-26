#!/usr/bin/env r
# Debug sampling to find off-by-one issue

library(torch)

# Simple test: create a tensor where position 6562 has 90% probability
# and see what index multinomial returns

cat("=== Testing torch_sort indexing ===\n")

# Create a simple probability distribution
vocab_size <- 6564L
probs <- torch::torch_zeros(1, vocab_size)
probs[1, 100] <- 0.05# R 1-indexed: position 100 = token 99 (0-indexed)
probs[1, 6563] <- 0.95# R 1-indexed: position 6563 = token 6562 (0-indexed)

cat(sprintf("Setting probs[1, 100] = 0.05 (token 99 in 0-indexed)\n"))
cat(sprintf("Setting probs[1, 6563] = 0.95 (token 6562 in 0-indexed)\n"))

# Sort descending
sorted_result <- torch::torch_sort(probs, descending = TRUE)
sorted_probs <- sorted_result[[1]]
sorted_indices <- sorted_result[[2]]

cat(sprintf("\nTop 3 sorted_indices (raw torch values):\n"))
cat(sprintf("  sorted_indices[1, 1] = %d\n", as.integer(sorted_indices[1, 1]$item())))
cat(sprintf("  sorted_indices[1, 2] = %d\n", as.integer(sorted_indices[1, 2]$item())))
cat(sprintf("  sorted_indices[1, 3] = %d\n", as.integer(sorted_indices[1, 3]$item())))

cat(sprintf("\nTop 3 sorted_probs:\n"))
cat(sprintf("  sorted_probs[1, 1] = %.4f\n", sorted_probs[1, 1]$item()))
cat(sprintf("  sorted_probs[1, 2] = %.4f\n", sorted_probs[1, 2]$item()))
cat(sprintf("  sorted_probs[1, 3] = %.4f\n", sorted_probs[1, 3]$item()))

# Now test multinomial
cat("\n=== Testing torch_multinomial ===\n")
torch::torch_manual_seed(42)
next_token_idx <- torch::torch_multinomial(sorted_probs, num_samples = 1L)
cat(sprintf("multinomial returned index: %d\n", as.integer(next_token_idx$item())))

# Gather
next_token <- sorted_indices$gather(2L, next_token_idx)
cat(sprintf("gathered token value: %d\n", as.integer(next_token$item())))

# Now let's verify the R indexing
cat("\n=== Verifying R tensor indexing ===\n")
test_tensor <- torch::torch_tensor(c(10, 20, 30, 40, 50))
cat(sprintf("test_tensor = [10, 20, 30, 40, 50]\n"))
cat(sprintf("test_tensor[1] = %d (R 1-indexed: first element)\n", as.integer(test_tensor[1]$item())))
cat(sprintf("test_tensor[2] = %d (R 1-indexed: second element)\n", as.integer(test_tensor[2]$item())))

# And let's check what index we need for token 6562
cat("\n=== Key finding ===\n")
if (as.integer(sorted_indices[1, 1]$item()) == 6562) {
    cat("sorted_indices[1, 1] = 6562: torch_sort returns 0-indexed values\n")
    cat("This means token 6562 is at position 1 (R 1-indexed) after sorting\n")
} else if (as.integer(sorted_indices[1, 1]$item()) == 6563) {
    cat("sorted_indices[1, 1] = 6563: torch_sort returns 1-indexed values!\n")
    cat("This might be the source of the off-by-one issue\n")
}

cat("\nDone.\n")

