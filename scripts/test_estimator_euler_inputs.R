#!/usr/bin/env Rscript
# Test estimator with Euler step inputs directly

library(chatterbox)

cat("Loading references...\n")
euler_ref <- chatterbox:::read_safetensors("outputs/euler_step_trace.safetensors")

# Load weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

estimator <- chatterbox:::cfm_estimator()
estimator$eval()
loaded <- chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
cat(sprintf("Loaded %d weight keys\n", loaded))

# Get the exact inputs from Python Euler step
x_in <- euler_ref$step0_x_in
mask_in <- torch::torch_ones(c(2, 1, 50)) # Python uses mask_in filled from mask
mu_in <- euler_ref$step0_mu_in
t_in <- euler_ref$step0_t_in
spks_in <- euler_ref$step0_spks_in
cond_in <- euler_ref$step0_cond_in

cat("\n=== Input comparison ===\n")
cat(sprintf("x_in: shape=%s, mean=%.6f\n", paste(dim(x_in), collapse = "x"), x_in$mean()$item()))
cat(sprintf("mu_in: shape=%s, mean=%.6f\n", paste(dim(mu_in), collapse = "x"), mu_in$mean()$item()))
cat(sprintf("t_in: shape=%s, values=%s\n", paste(dim(t_in), collapse = "x"),
        paste(sprintf("%.6f", as.numeric(t_in)), collapse = ", ")))
cat(sprintf("spks_in: shape=%s, mean=%.6f\n", paste(dim(spks_in), collapse = "x"), spks_in$mean()$item()))
cat(sprintf("cond_in: shape=%s, mean=%.6f\n", paste(dim(cond_in), collapse = "x"), cond_in$mean()$item()))

cat("\n=== Estimator forward ===\n")
torch::with_no_grad({
        # Forward pass
        output <- estimator$forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

        cat(sprintf("R output: mean=%.6f, std=%.6f\n", output$mean()$item(), output$std()$item()))
        cat(sprintf("Py output: mean=%.6f, std=%.6f\n",
                euler_ref$step0_dphi_raw$mean()$item(), euler_ref$step0_dphi_raw$std()$item()))

        diff <- (output - euler_ref$step0_dphi_raw)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))

        # Check batch items separately
        cat(sprintf("\nBatch 0 (conditional):\n"))
        cat(sprintf("  R: mean=%.6f, Py: mean=%.6f\n",
                output[1,,]$mean()$item(), euler_ref$step0_dphi_raw[1,,]$mean()$item()))

        cat(sprintf("Batch 1 (unconditional):\n"))
        cat(sprintf("  R: mean=%.6f, Py: mean=%.6f\n",
                output[2,,]$mean()$item(), euler_ref$step0_dphi_raw[2,,]$mean()$item()))

        # Check if any specific part matches better
        cat("\n=== Checking intermediate ===\n")

        # Pack inputs manually (same as forward)
        time <- 50L
        h <- torch::torch_cat(list(x_in, mu_in), dim = 2L)
        spks_exp <- spks_in$unsqueeze(3L)$expand(c(- 1L, - 1L, time))
        h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
        h <- torch::torch_cat(list(h, cond_in), dim = 2L)

        cat(sprintf("Packed inputs: shape=%s, mean=%.6f\n",
                paste(dim(h), collapse = "x"), h$mean()$item()))
    })

