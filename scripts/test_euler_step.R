#!/usr/bin/env Rscript
# Compare Euler step with Python reference

library(chatterbox)

cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/euler_step_trace.safetensors")

# Load weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

decoder <- chatterbox:::causal_cfm()
decoder$eval()
chatterbox:::load_cfm_estimator_weights(decoder$estimator, state_dict, prefix = "flow.decoder.estimator.")

# Get inputs from reference
mu <- ref$mu
mask <- ref$mask
spks <- ref$spks
cond <- ref$cond

time <- mu$size(3)

cat("\n=== Comparing Euler step ===\n")
torch::with_no_grad({
        # Copy Python's initial noise
        z <- ref$z_initial

        cat(sprintf("Initial noise: R z mean=%.6f (Py: %.6f)\n",
                z$mean()$item(), ref$z_initial$mean()$item()))

        # Time schedule
        t_span <- torch::torch_linspace(0, 1, 11, dtype = mu$dtype)
        t_span <- 1 - torch::torch_cos(t_span * 0.5 * pi)

        cat(sprintf("t_span[1]: R=%.6f, Py=%.6f\n",
                as.numeric(t_span[1]), as.numeric(ref$t_span[1])))
        cat(sprintf("t_span[2]: R=%.6f, Py=%.6f\n",
                as.numeric(t_span[2]), as.numeric(ref$t_span[2])))

        t <- t_span[1]$unsqueeze(1)
        dt <- t_span[2] - t_span[1]

        cat(sprintf("t: %.6f, dt: %.6f\n", t$item(), dt$item()))

        # Setup CFG inputs
        x_in <- torch::torch_zeros(c(2, 80, time), dtype = mu$dtype)
        mask_in <- torch::torch_zeros(c(2, 1, time), dtype = mu$dtype)
        mu_in <- torch::torch_zeros(c(2, 80, time), dtype = mu$dtype)
        t_in <- torch::torch_zeros(2, dtype = mu$dtype)
        spks_in <- torch::torch_zeros(c(2, 80), dtype = mu$dtype)
        cond_in <- torch::torch_zeros(c(2, 80, time), dtype = mu$dtype)

        x <- z$clone()

        # First step
        x_in[1:2,,] <- x
        mask_in[1:2,,] <- mask
        mu_in[1,,] <- mu
        t_in[1:2] <- t
        spks_in[1,] <- spks
        cond_in[1,,] <- cond

        cat(sprintf("\nStep 0:\n"))
        cat(sprintf("  x_in: R mean=%.6f (Py: %.6f)\n",
                x_in$mean()$item(), ref$step0_x_in$mean()$item()))
        cat(sprintf("  mu_in: R mean=%.6f (Py: %.6f)\n",
                mu_in$mean()$item(), ref$step0_mu_in$mean()$item()))

        # Check x_in diff
        diff <- (x_in - ref$step0_x_in)$abs()$max()$item()
        cat(sprintf("  x_in diff: %.6f\n", diff))

        # Forward through estimator
        dphi_dt <- decoder$estimator$forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

        cat(sprintf("  dphi_dt (raw): R mean=%.6f (Py: %.6f)\n",
                dphi_dt$mean()$item(), ref$step0_dphi_raw$mean()$item()))

        diff <- (dphi_dt - ref$step0_dphi_raw)$abs()$max()$item()
        cat(sprintf("  dphi_raw diff: %.6f\n", diff))

        # CFG combination - R indexing starts at 1
        dphi_cond <- dphi_dt[1,,]$unsqueeze(1)
        dphi_uncond <- dphi_dt[2,,]$unsqueeze(1)

        cat(sprintf("  dphi_cond: R mean=%.6f (Py: %.6f)\n",
                dphi_cond$mean()$item(), ref$step0_dphi_cond$mean()$item()))
        cat(sprintf("  dphi_uncond: R mean=%.6f (Py: %.6f)\n",
                dphi_uncond$mean()$item(), ref$step0_dphi_uncond$mean()$item()))

        cfg_rate <- decoder$inference_cfg_rate
        dphi_combined <- (1.0 + cfg_rate) * dphi_cond - cfg_rate * dphi_uncond

        cat(sprintf("  cfg_rate: %.1f\n", cfg_rate))
        cat(sprintf("  dphi_combined: R mean=%.6f (Py: %.6f)\n",
                dphi_combined$mean()$item(), ref$step0_dphi_combined$mean()$item()))

        diff <- (dphi_combined - ref$step0_dphi_combined)$abs()$max()$item()
        cat(sprintf("  dphi_combined diff: %.6f\n", diff))

        # Euler step
        x <- x + dt * dphi_combined

        cat(sprintf("  x after step: R mean=%.6f (Py: %.6f)\n",
                x$mean()$item(), ref$step1_x$mean()$item()))

        diff <- (x - ref$step1_x)$abs()$max()$item()
        cat(sprintf("  step1_x diff: %.6f\n", diff))

        # Full decoder result
        cat(sprintf("\nFull result: Py mean=%.6f, std=%.6f\n",
                ref$full_result$mean()$item(), ref$full_result$std()$item()))
    })

