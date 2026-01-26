#!/usr/bin/env Rscript
# Check feed-forward mechanism in CFM transformer

library(chatterbox)

cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

# Check what keys exist for FF
cat("=== FF weight keys in state dict ===\n")
ff_keys <- grep("mid_blocks.0.1.0.ff", names(state_dict), value = TRUE)
for (k in ff_keys) {
    cat(sprintf("%s: %s\n", k, paste(dim(state_dict[[k]]), collapse = "x")))
}

estimator <- chatterbox:::cfm_estimator()
chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
estimator$eval()

tfm <- estimator$mid_transformers[[1]][[1]]

cat("\n=== R FF structure ===\n")
cat(sprintf("ff$net[[1]] (GELU with proj): %s\n", class(tfm$ff$net[[1]])[1]))
cat(sprintf("ff$net[[1]]$proj weight: %s\n", paste(dim(tfm$ff$net[[1]]$proj$weight), collapse = "x")))
cat(sprintf("ff$net[[3]] (output linear) weight: %s\n", paste(dim(tfm$ff$net[[3]]$weight), collapse = "x")))

# Test FF forward
torch::with_no_grad({
        hidden <- torch::torch_randn(c(1L, 50L, 256L)) * 0.1
        cat("\n=== Testing FF forward ===\n")
        cat(sprintf("Input: mean=%.4f, std=%.4f\n", hidden$mean()$item(), hidden$std()$item()))

        # Step through FF
        h <- tfm$ff$net[[1]]$forward(hidden)
        cat(sprintf("After GELU proj: shape=%s, mean=%.4f, std=%.4f\n",
                paste(dim(h), collapse = "x"), h$mean()$item(), h$std()$item()))

        h <- tfm$ff$net[[3]]$forward(h)
        cat(sprintf("After output linear: shape=%s, mean=%.4f, std=%.4f\n",
                paste(dim(h), collapse = "x"), h$mean()$item(), h$std()$item()))

        # Full FF
        h_full <- tfm$ff$forward(hidden)
        cat(sprintf("Full FF output: mean=%.4f, std=%.4f\n", h_full$mean()$item(), h_full$std()$item()))
    })

# Now check running through multiple transformer blocks
cat("\n=== Running through 4 transformer blocks ===\n")
torch::with_no_grad({
        h <- torch::torch_randn(c(1L, 50L, 256L)) * 0.3# Similar to actual data
        t_emb <- torch::torch_randn(c(1L, 1024L))

        cat(sprintf("Initial: mean=%.4f, std=%.4f\n", h$mean()$item(), h$std()$item()))

        for (i in 1:4) {
            h <- estimator$mid_transformers[[1]][[i]]$forward(h, NULL, t_emb)
            cat(sprintf("After tfm %d: mean=%.4f, std=%.4f\n", i, h$mean()$item(), h$std()$item()))
        }
    })

