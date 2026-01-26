#!/usr/bin/env Rscript
# Test conformer encoder with loaded weights against Python reference

library(chatterbox)

# Load Python reference
cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/encoder_reference.safetensors")

cat(sprintf("Python reference:\n"))
cat(sprintf("  input: %s, mean=%.6f, std=%.6f\n",
        paste(dim(ref$input), collapse = "x"),
        ref$input$mean()$item(), ref$input$std()$item()))
cat(sprintf("  output: %s, mean=%.6f, std=%.6f\n",
        paste(dim(ref$output), collapse = "x"),
        ref$output$mean()$item(), ref$output$std()$item()))

# Load s3gen weights
cat("\nLoading s3gen weights...\n")
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
if (!file.exists(weights_path)) {
    stop("s3gen.safetensors not found. Run load_chatterbox() first.")
}
weights <- chatterbox:::read_safetensors(weights_path)

# Create encoder
cat("Creating encoder...\n")
encoder <- chatterbox:::upsample_conformer_encoder_full(
    input_size = 512L,
    output_size = 512L,
    num_blocks = 6L,
    num_up_blocks = 4L,
    n_head = 8L,
    n_ffn = 2048L
)

# Load weights
cat("Loading encoder weights...\n")
chatterbox:::load_conformer_encoder_weights(encoder, weights, prefix = "flow.encoder.")

# Set to eval mode
encoder$eval()

# Run with Python input
cat("\nRunning encoder forward...\n")
test_input <- ref$input
test_lens <- torch::torch_tensor(50L)$view(c(1L))

# Forward
torch::with_no_grad({
        result <- encoder$forward(test_input, test_lens)
        output <- result[[1]]
        output_lens <- result[[2]]

        cat(sprintf("\nR output:\n"))
        cat(sprintf("  shape: %s\n", paste(dim(output), collapse = "x")))
        cat(sprintf("  mean: %.6f\n", output$mean()$item()))
        cat(sprintf("  std: %.6f\n", output$std()$item()))

        # Compare
        diff <- (output - ref$output)$abs()$max()$item()
        cat(sprintf("\nMax diff: %.6f\n", diff))

        # More detailed comparison
        mean_diff <- abs(output$mean()$item() - ref$output$mean()$item())
        std_diff <- abs(output$std()$item() - ref$output$std()$item())
        cat(sprintf("Mean diff: %.6f\n", mean_diff))
        cat(sprintf("Std diff: %.6f\n", std_diff))
    })

