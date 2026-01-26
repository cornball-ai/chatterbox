#!/usr/bin/env Rscript
# Test CFM estimator with loaded weights against Python reference

library(chatterbox)

cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")

# Load s3gen weights
cat("Loading s3gen weights...\n")
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
if (!file.exists(weights_path)) {
    stop("Weights not found at: ", weights_path)
}
state_dict <- chatterbox:::read_safetensors(weights_path)

# Create estimator
cat("Creating CFM estimator...\n")
estimator <- chatterbox:::cfm_estimator()

# Load weights
cat("Loading weights...\n")
loaded <- chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
cat(sprintf("Loaded %d weight keys\n", loaded))

# Get inputs from reference
x <- ref$input_x
mask <- ref$input_mask
mu <- ref$input_mu
t <- ref$input_t
spks <- ref$input_spks
cond <- ref$input_cond

cat("\n=== Testing forward pass ===\n")
estimator$eval()
torch::with_no_grad({
        output <- estimator$forward(x, mask, mu, t, spks, cond)
        cat(sprintf("R output: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(output), collapse = "x"),
                output$mean()$item(),
                output$std()$item()))
        cat(sprintf("Py output: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(ref$full_output), collapse = "x"),
                ref$full_output$mean()$item(),
                ref$full_output$std()$item()))

        diff <- (output - ref$full_output)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))

        if (diff < 0.01) {
            cat("\nVALIDATION PASSED\n")
        } else {
            cat("\nVALIDATION FAILED - checking intermediate values...\n")

            # Check time embedding
            t_emb_sin <- estimator$time_embeddings$forward(t)
            t_emb <- estimator$time_mlp$forward(t_emb_sin)

            diff_time <- (t_emb - ref$time_emb_mlp)$abs()$max()$item()
            cat(sprintf("  time_emb diff: %.6f\n", diff_time))
        }
    })

