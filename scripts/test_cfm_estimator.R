#!/usr/bin/env Rscript
# Test CFM estimator against Python reference

library(chatterbox)

cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")

# Create estimator
cat("Creating CFM estimator...\n")
estimator <- chatterbox:::cfm_estimator()

cat(sprintf("Parameters: %d\n", sum(sapply(estimator$parameters, function (p) prod(dim(p))))))

# Get inputs from reference
x <- ref$input_x
mask <- ref$input_mask
mu <- ref$input_mu
t <- ref$input_t
spks <- ref$input_spks
cond <- ref$input_cond

cat("\nInput shapes:\n")
cat(sprintf("  x: %s\n", paste(dim(x), collapse = "x")))
cat(sprintf("  mask: %s\n", paste(dim(mask), collapse = "x")))
cat(sprintf("  mu: %s\n", paste(dim(mu), collapse = "x")))
cat(sprintf("  t: %s\n", paste(dim(t), collapse = "x")))
cat(sprintf("  spks: %s\n", paste(dim(spks), collapse = "x")))
cat(sprintf("  cond: %s\n", paste(dim(cond), collapse = "x")))

# Test forward pass (with random weights)
cat("\nTesting forward pass with random weights...\n")
torch::with_no_grad({
        output <- estimator$forward(x, mask, mu, t, spks, cond)
        cat(sprintf("Output shape: %s\n", paste(dim(output), collapse = "x")))
        cat(sprintf("Output mean: %.6f, std: %.6f\n", output$mean()$item(), output$std()$item()))
    })

# Compare time embedding
cat("\n=== Comparing time embeddings ===\n")
torch::with_no_grad({
        t_emb_sin <- estimator$time_embeddings$forward(t)
        cat(sprintf("R time_emb_sinusoidal: shape=%s, mean=%.6f\n",
                paste(dim(t_emb_sin), collapse = "x"), t_emb_sin$mean()$item()))
        cat(sprintf("Py time_emb_sinusoidal: shape=%s, mean=%.6f\n",
                paste(dim(ref$time_emb_sinusoidal), collapse = "x"), ref$time_emb_sinusoidal$mean()$item()))

        diff <- (t_emb_sin - ref$time_emb_sinusoidal)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))
    })

# Compare packed inputs
cat("\n=== Comparing input packing ===\n")
torch::with_no_grad({
        # x + mu
        packed_x_mu <- torch::torch_cat(list(x, mu), dim = 2L)
        cat(sprintf("R packed_x_mu: shape=%s, mean=%.6f\n",
                paste(dim(packed_x_mu), collapse = "x"), packed_x_mu$mean()$item()))
        cat(sprintf("Py packed_x_mu: shape=%s, mean=%.6f\n",
                paste(dim(ref$packed_x_mu), collapse = "x"), ref$packed_x_mu$mean()$item()))

        diff <- (packed_x_mu - ref$packed_x_mu)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))

        # + spks
        spks_exp <- spks$unsqueeze(3L)$expand(c(- 1L, - 1L, 50L))
        packed_spks <- torch::torch_cat(list(packed_x_mu, spks_exp), dim = 2L)
        cat(sprintf("\nR packed_spks: shape=%s, mean=%.6f\n",
                paste(dim(packed_spks), collapse = "x"), packed_spks$mean()$item()))
        cat(sprintf("Py packed_spks: shape=%s, mean=%.6f\n",
                paste(dim(ref$packed_spks), collapse = "x"), ref$packed_spks$mean()$item()))

        diff <- (packed_spks - ref$packed_spks)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))

        # + cond
        packed_all <- torch::torch_cat(list(packed_spks, cond), dim = 2L)
        cat(sprintf("\nR packed_all: shape=%s, mean=%.6f\n",
                paste(dim(packed_all), collapse = "x"), packed_all$mean()$item()))
        cat(sprintf("Py packed_all: shape=%s, mean=%.6f\n",
                paste(dim(ref$packed_all), collapse = "x"), ref$packed_all$mean()$item()))

        diff <- (packed_all - ref$packed_all)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))
    })

cat("\nTest complete.\n")

