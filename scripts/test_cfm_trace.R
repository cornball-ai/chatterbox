#!/usr/bin/env Rscript
# Detailed CFM estimator tracing

library(chatterbox)

cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")

# Load s3gen weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

# Create estimator and load weights
estimator <- chatterbox:::cfm_estimator()
estimator$eval()
loaded <- chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
cat(sprintf("Loaded %d weight keys\n", loaded))

# Get inputs
x <- ref$input_x
mask <- ref$input_mask
mu <- ref$input_mu
t <- ref$input_t
spks <- ref$input_spks
cond <- ref$input_cond

cat("\n=== Full forward trace ===\n")
torch::with_no_grad({
        batch_size <- x$size(1)
        seq_len <- x$size(3)

        # Time embedding
        t_emb <- estimator$time_embeddings$forward(t)$to(dtype = t$dtype)
        t_emb <- estimator$time_mlp$forward(t_emb)

        # Pack inputs
        h <- torch::torch_cat(list(x, mu), dim = 2L)
        spks_exp <- spks$unsqueeze(3L)$expand(c(- 1L, - 1L, seq_len))
        h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
        h <- torch::torch_cat(list(h, cond), dim = 2L)

        # Down block
        h <- estimator$down_resnet$forward(h, mask, t_emb)
        h <- h$transpose(2L, 3L)$contiguous()
        attn_mask <- NULL
        for (i in seq_along(estimator$down_transformers)) {
            h <- estimator$down_transformers[[i]]$forward(h, attn_mask, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous()
        hidden_skip <- h
        h <- estimator$down_conv$forward(h * mask)

        # Compare all mid blocks
        for (i in 1:12) {
            h <- estimator$mid_resnets[[i]]$forward(h, mask, t_emb)
            h <- h$transpose(2L, 3L)$contiguous()
            for (j in seq_along(estimator$mid_transformers[[i]])) {
                h <- estimator$mid_transformers[[i]][[j]]$forward(h, attn_mask, t_emb)
            }
            h <- h$transpose(2L, 3L)$contiguous()

            ref_name <- paste0("mid_", i - 1, "_tfm_0")
            if (ref_name %in% names(ref)) {
                ref_tfm <- ref[[ref_name]]
                # Note: ref is (B, T, C), h is (B, C, T)
                # Need to transpose to compare
            }

            ref_resnet_name <- paste0("mid_", i - 1, "_resnet")
            if (ref_resnet_name %in% names(ref)) {
                ref_res <- ref[[ref_resnet_name]]
                # ref is (B, C, T) for resnet
            }
        }

        cat(sprintf("After all mid blocks: mean=%.6f, std=%.6f\n", h$mean()$item(), h$std()$item()))
        cat(sprintf("Python mid_11_resnet: mean=%.6f, std=%.6f\n", ref$mid_11_resnet$mean()$item(), ref$mid_11_resnet$std()$item()))

        # Up block
        h <- estimator$up_conv$forward(h * mask)
        cat(sprintf("After up_conv: mean=%.6f, std=%.6f\n", h$mean()$item(), h$std()$item()))
        cat(sprintf("Python upsample: mean=%.6f, std=%.6f\n", ref$upsample$mean()$item(), ref$upsample$std()$item()))

        diff <- (h - ref$upsample)$abs()$max()$item()
        cat(sprintf("  up_conv diff: %.6f\n", diff))

        h <- torch::torch_cat(list(h, hidden_skip), dim = 2L)
        h <- estimator$up_resnet$forward(h, mask, t_emb)
        cat(sprintf("After up_resnet: mean=%.6f, std=%.6f\n", h$mean()$item(), h$std()$item()))
        cat(sprintf("Python up_resnet: mean=%.6f, std=%.6f\n", ref$up_resnet$mean()$item(), ref$up_resnet$std()$item()))

        diff <- (h - ref$up_resnet)$abs()$max()$item()
        cat(sprintf("  up_resnet diff: %.6f\n", diff))

        h <- h$transpose(2L, 3L)$contiguous()
        for (i in seq_along(estimator$up_transformers)) {
            h <- estimator$up_transformers[[i]]$forward(h, attn_mask, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous()
        cat(sprintf("After up_transformers: mean=%.6f, std=%.6f\n", h$mean()$item(), h$std()$item()))

        # Final
        h <- estimator$final_block$forward(h, mask)
        cat(sprintf("After final_block: mean=%.6f, std=%.6f\n", h$mean()$item(), h$std()$item()))
        cat(sprintf("Python final_block: mean=%.6f, std=%.6f\n", ref$final_block$mean()$item(), ref$final_block$std()$item()))

        diff <- (h - ref$final_block)$abs()$max()$item()
        cat(sprintf("  final_block diff: %.6f\n", diff))

        h <- estimator$final_proj$forward(h * mask)
        cat(sprintf("After final_proj: mean=%.6f, std=%.6f\n", h$mean()$item(), h$std()$item()))
        cat(sprintf("Python final_proj: mean=%.6f, std=%.6f\n", ref$final_proj$mean()$item(), ref$final_proj$std()$item()))

        diff <- (h - ref$final_proj)$abs()$max()$item()
        cat(sprintf("  final_proj diff: %.6f\n", diff))
    })

