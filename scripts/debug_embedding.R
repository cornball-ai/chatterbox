#!/usr/bin/env r
# Debug embedding indexing

library(torch)

cat("=== Testing nn_embedding indexing ===\n")

# Create a small embedding with known values
emb <- torch::nn_embedding(5, 3)

# Set known values for testing
torch::with_no_grad({
        emb$weight[1,] <- torch::torch_tensor(c(1.0, 1.0, 1.0))
        emb$weight[2,] <- torch::torch_tensor(c(2.0, 2.0, 2.0))
        emb$weight[3,] <- torch::torch_tensor(c(3.0, 3.0, 3.0))
        emb$weight[4,] <- torch::torch_tensor(c(4.0, 4.0, 4.0))
        emb$weight[5,] <- torch::torch_tensor(c(5.0, 5.0, 5.0))
    })

cat("Embedding weights:\n")
cat("  weight[1,] = c(1,1,1) - R index 1\n")
cat("  weight[2,] = c(2,2,2) - R index 2\n")
cat("  ...\n")

# Test forward with different inputs
cat("\n=== Testing forward() ===\n")

# Input 0 (as a tensor)
input0 <- torch::torch_tensor(matrix(0L, nrow = 1), dtype = torch::torch_long())
result0 <- emb$forward(input0)
cat(sprintf("forward(0) = [%.0f, %.0f, %.0f]\n", result0[1, 1, 1]$item(), result0[1, 1, 2]$item(), result0[1, 1, 3]$item()))

# Input 1
input1 <- torch::torch_tensor(matrix(1L, nrow = 1), dtype = torch::torch_long())
result1 <- emb$forward(input1)
cat(sprintf("forward(1) = [%.0f, %.0f, %.0f]\n", result1[1, 1, 1]$item(), result1[1, 1, 2]$item(), result1[1, 1, 3]$item()))

# Input 2
input2 <- torch::torch_tensor(matrix(2L, nrow = 1), dtype = torch::torch_long())
result2 <- emb$forward(input2)
cat(sprintf("forward(2) = [%.0f, %.0f, %.0f]\n", result2[1, 1, 1]$item(), result2[1, 1, 2]$item(), result2[1, 1, 3]$item()))

cat("\n=== Key finding ===\n")
if (result0[1, 1, 1]$item() == 1.0) {
    cat("forward(0) = weight[1,]: nn_embedding uses 0-indexed input -> 1-indexed weight\n")
    cat("Input 0 gets weight at R position 1\n")
} else if (result1[1, 1, 1]$item() == 1.0) {
    cat("forward(1) = weight[1,]: nn_embedding uses 1-indexed input\n")
}

cat("\nDone.\n")

