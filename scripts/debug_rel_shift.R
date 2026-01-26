#!/usr/bin/env Rscript
# Debug rel_shift function

library(torch)

# Simulate the rel_shift function
rel_shift <- function (x)
{
    # x: (batch, head, time, 2*time-1)
    # Shift to align relative positions
    batch_size <- x$size(1)
    n_head <- x$size(2)
    seq_len <- x$size(3)
    pos_len <- x$size(4)

    cat("rel_shift input: batch=", batch_size, "head=", n_head,
        "seq=", seq_len, "pos=", pos_len, "\n")

    # Pad left side with 1 zero
    x <- torch::nnf_pad(x, c(1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L))
    cat("After pad:", paste(x$shape, collapse = "x"), "\n")

    # Reshape and slice to get relative positions
    x <- x$view(c(batch_size, n_head, pos_len + 1, seq_len))
    cat("After view:", paste(x$shape, collapse = "x"), "\n")

    x <- x[,, 2:(seq_len + 1),]$contiguous()
    cat("After slice:", paste(x$shape, collapse = "x"), "\n")

    x <- x$view(c(batch_size, n_head, seq_len, seq_len))
    cat("Final:", paste(x$shape, collapse = "x"), "\n")

    x
}

# Test with expected dimensions
# For seq_len=32, matrix_bd should be (1, 8, 32, 63) where 63 = 2*32-1
x <- torch::torch_randn(c(1, 8, 32, 63))
cat("\nTest input shape:", paste(x$shape, collapse = "x"), "\n\n")

tryCatch({
        result <- rel_shift(x)
        cat("\nResult shape:", paste(result$shape, collapse = "x"), "\n")
    }, error = function (e)
    {
        cat("\nError:", e$message, "\n")
    })

# Check the Python logic
cat("\n=== Python logic analysis ===\n")
cat("For seq_len=32, pos_len=63:\n")
cat("After pad: shape (1, 8, 32, 64)\n")
cat("After view as (B, H, pos_len+1, seq_len): (1, 8, 64, 32)\n")
cat("After slice [2:seq_len+1]: (1, 8, 32, 32)\n")
cat("After view as (B, H, seq_len, seq_len): (1, 8, 32, 32)\n")

