#!/usr/bin/env Rscript
# Test full CFM decoder (10 Euler steps)

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
z_initial <- ref$z_initial

cat("\n=== Running full CFM decoder ===\n")
cat(sprintf("Inputs: mu=%s, mask=%s, spks=%s, cond=%s\n",
        paste(dim(mu), collapse = "x"),
        paste(dim(mask), collapse = "x"),
        paste(dim(spks), collapse = "x"),
        paste(dim(cond), collapse = "x")))

torch::with_no_grad({
        # Build t_span same as Python (10 steps, cosine schedule)
        t_span <- torch::torch_linspace(0, 1, 11, dtype = mu$dtype)
        t_span <- 1 - torch::torch_cos(t_span * 0.5 * pi)

        # Run full solve_euler with the same initial noise
        result <- decoder$solve_euler(
            x = z_initial,
            t_span = t_span,
            mu = mu,
            mask = mask,
            spks = spks,
            cond = cond
        )

        cat(sprintf("\nR result: mean=%.6f, std=%.6f\n",
                result$mean()$item(), result$std()$item()))
        cat(sprintf("Py result: mean=%.6f, std=%.6f\n",
                ref$full_result$mean()$item(), ref$full_result$std()$item()))

        diff <- (result - ref$full_result)$abs()$max()$item()
        cat(sprintf("\nMax diff: %.6f\n", diff))

        if (diff < 1.0) {
            cat("\n=== CFM DECODER VALIDATED ===\n")
        } else {
            cat("\n=== CFM DECODER NEEDS WORK ===\n")
        }
    })

