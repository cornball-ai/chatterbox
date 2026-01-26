#!/usr/bin/env Rscript
# Compare transformer block outputs in mid blocks

library(chatterbox)

cat("Loading references...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")

cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

estimator <- chatterbox:::cfm_estimator()
estimator$eval()
chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")

# Get inputs
x <- ref$input_x
mask <- ref$input_mask
mu <- ref$input_mu
t <- ref$input_t
spks <- ref$input_spks
cond <- ref$input_cond

cat("\n=== Transformer comparison ===\n")
torch::with_no_grad({
        batch_size <- x$size(1)
        seq_len <- x$size(3)

        # Setup
        t_emb <- estimator$time_embeddings$forward(t)$to(dtype = t$dtype)
        t_emb <- estimator$time_mlp$forward(t_emb)

        h <- torch::torch_cat(list(x, mu), dim = 2L)
        spks_exp <- spks$unsqueeze(3L)$expand(c(- 1L, - 1L, seq_len))
        h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
        h <- torch::torch_cat(list(h, cond), dim = 2L)

        # Down block
        h <- estimator$down_resnet$forward(h, mask, t_emb)
        h <- h$transpose(2L, 3L)$contiguous()
        for (i in seq_along(estimator$down_transformers)) {
            h <- estimator$down_transformers[[i]]$forward(h, NULL, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous()
        h <- estimator$down_conv$forward(h * mask)

        # Mid block 0 - check both resnet and transformers
        cat("=== Mid block 0 ===\n")
        h <- estimator$mid_resnets[[1]]$forward(h, mask, t_emb)
        cat(sprintf("After resnet: mean=%.4f (Py: %.4f)\n",
                h$mean()$item(), ref$mid_0_resnet$mean()$item()))

        h <- h$transpose(2L, 3L)$contiguous()
        cat(sprintf("Before transformers (B,T,C): shape=%s, mean=%.4f\n",
                paste(dim(h), collapse = "x"), h$mean()$item()))

        # Run first transformer only
        h_tfm0 <- estimator$mid_transformers[[1]][[1]]$forward(h, NULL, t_emb)
        cat(sprintf("After tfm_0: mean=%.4f (Py: %.4f)\n",
                h_tfm0$mean()$item(), ref$mid_0_tfm_0$mean()$item()))
        diff <- (h_tfm0 - ref$mid_0_tfm_0)$abs()$max()$item()
        cat(sprintf("  diff: %.6f\n", diff))

        # Check if there's drift through all 4 transformers
        h_running <- h
        for (j in 1:4) {
            h_running <- estimator$mid_transformers[[1]][[j]]$forward(h_running, NULL, t_emb)
        }
        cat(sprintf("After all 4 tfms: mean=%.4f, std=%.4f\n",
                h_running$mean()$item(), h_running$std()$item()))
        h_running <- h_running$transpose(2L, 3L)$contiguous()

        # Continue to mid block 1
        cat("\n=== Mid block 1 ===\n")
        h_running <- estimator$mid_resnets[[2]]$forward(h_running, mask, t_emb)
        cat(sprintf("After resnet: mean=%.4f (Py: %.4f)\n",
                h_running$mean()$item(), ref$mid_1_resnet$mean()$item()))
        diff <- (h_running - ref$mid_1_resnet)$abs()$max()$item()
        cat(sprintf("  resnet diff: %.6f\n", diff))

        h_running <- h_running$transpose(2L, 3L)$contiguous()
        for (j in 1:4) {
            h_running <- estimator$mid_transformers[[2]][[j]]$forward(h_running, NULL, t_emb)
        }
        h_running <- h_running$transpose(2L, 3L)$contiguous()

        # Check mid_1_tfm_0
        cat(sprintf("After tfms (transposed back): mean=%.4f\n", h_running$mean()$item()))

        # Let's check if the tfm_0 reference is (B,T,C) or (B,C,T)
        cat(sprintf("\nRef mid_0_tfm_0 shape: %s\n", paste(dim(ref$mid_0_tfm_0), collapse = "x")))
        cat(sprintf("Ref mid_1_resnet shape: %s\n", paste(dim(ref$mid_1_resnet), collapse = "x")))
    })

