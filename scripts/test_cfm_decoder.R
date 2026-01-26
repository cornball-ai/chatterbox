#!/usr/bin/env Rscript
# Test full CFM decoder (causal_cfm) against Python

library(chatterbox)

cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_decoder_output.safetensors")

# Load s3gen weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

# Create decoder
cat("Creating CFM decoder...\n")
decoder <- chatterbox:::causal_cfm()
decoder$eval()

# Load estimator weights
chatterbox:::load_cfm_estimator_weights(decoder$estimator, state_dict, prefix = "flow.decoder.estimator.")

# Copy the rand_noise from Python reference
cat("Copying Python rand_noise...\n")
torch::with_no_grad({
        # Expand python noise to match our buffer size
        py_noise <- ref$rand_noise
        cat(sprintf("Python noise: shape=%s, mean=%.6f\n",
                paste(dim(py_noise), collapse = "x"), py_noise$mean()$item()))

        # Our buffer is larger (50*300), copy the first 50 timesteps
        decoder$rand_noise[,, 1:50]$copy_(py_noise)
    })

# Get inputs from reference
mu <- ref$mu
mask <- ref$mask
spks <- ref$spks
cond <- ref$cond

cat("\n=== Testing decoder forward ===\n")
torch::with_no_grad({
        result <- decoder$forward(mu, mask, spks, cond, n_timesteps = 10L)
        output <- result[[1]]

        cat(sprintf("R output: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(output), collapse = "x"),
                output$mean()$item(),
                output$std()$item()))

        cat(sprintf("Py output: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(ref$decoder_output), collapse = "x"),
                ref$decoder_output$mean()$item(),
                ref$decoder_output$std()$item()))

        diff <- (output - ref$decoder_output)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))

        if (diff < 0.5) {
            cat("\nVALIDATION PASSED\n")
        } else {
            cat("\nVALIDATION FAILED - checking Euler solver...\n")

            # Debug: Check single estimator forward
            seq_len <- mu$size(3)
            z <- decoder$rand_noise[,, 1:seq_len]$to(dtype = mu$dtype)

            # Check time span
            t_span <- torch::torch_linspace(0, 1, 11, dtype = mu$dtype)
            t_span <- 1 - torch::torch_cos(t_span * 0.5 * pi)
            cat(sprintf("Time span: %s\n", paste(sprintf("%.4f", as.numeric(t_span)), collapse = ", ")))

            # Single estimator forward at t=0
            t <- t_span[1]$unsqueeze(1)

            # CFG batch (conditional + unconditional)
            x_in <- torch::torch_zeros(c(2, 80, seq_len), dtype = mu$dtype)
            x_in[1:2,,] <- z
            mask_in <- torch::torch_zeros(c(2, 1, seq_len), dtype = mu$dtype)
            mask_in[1:2,,] <- mask
            mu_in <- torch::torch_zeros(c(2, 80, seq_len), dtype = mu$dtype)
            mu_in[1,,] <- mu
            t_in <- torch::torch_zeros(2, dtype = mu$dtype)
            t_in[1:2] <- t
            spks_in <- torch::torch_zeros(c(2, 80), dtype = mu$dtype)
            spks_in[1,] <- spks
            cond_in <- torch::torch_zeros(c(2, 80, seq_len), dtype = mu$dtype)
            cond_in[1,,] <- cond

            dphi <- decoder$estimator$forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            cat(sprintf("Estimator output at t=0: mean=%.6f, std=%.6f\n",
                    dphi$mean()$item(), dphi$std()$item()))
        }
    })

