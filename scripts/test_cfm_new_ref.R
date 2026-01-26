#!/usr/bin/env Rscript
# Compare CFM estimator to new Python reference (with attention mask = 0)

library(chatterbox)

cat("Loading new Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_mid_details.safetensors")

# Load weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

estimator <- chatterbox:::cfm_estimator()
estimator$eval()
chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")

# Use same inputs as original cfm_estimator_steps
old_ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")
x <- old_ref$input_x
mask <- old_ref$input_mask
mu <- old_ref$input_mu
t <- old_ref$input_t
spks <- old_ref$input_spks
cond <- old_ref$input_cond

cat("\n=== Comparing key intermediate values ===\n")
torch::with_no_grad({
        # Time embedding
        t_emb <- estimator$time_embeddings$forward(t)$to(dtype = t$dtype)
        t_emb <- estimator$time_mlp$forward(t_emb)
        diff <- (t_emb - ref$time_emb)$abs()$max()$item()
        cat(sprintf("time_emb: diff=%.6f\n", diff))

        # Full forward
        output <- estimator$forward(x, mask, mu, t, spks, cond)
        cat(sprintf("\nR output: mean=%.6f, std=%.6f\n", output$mean()$item(), output$std()$item()))
        cat(sprintf("Py output: mean=%.6f, std=%.6f\n", ref$final_proj$mean()$item(), ref$final_proj$std()$item()))

        diff <- (output - ref$final_proj)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))

        if (diff < 0.1) {
            cat("\nVALIDATION PASSED\n")
        } else {
            cat("\nDiff > 0.1, checking intermediate values...\n")

            # Check each mid block
            batch_size <- x$size(1)
            seq_len <- x$size(3)

            h <- torch::torch_cat(list(x, mu), dim = 2L)
            spks_exp <- spks$unsqueeze(3L)$expand(c(- 1L, - 1L, seq_len))
            h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
            h <- torch::torch_cat(list(h, cond), dim = 2L)

            h <- estimator$down_resnet$forward(h, mask, t_emb)
            diff <- (h - ref$down_resnet)$abs()$max()$item()
            cat(sprintf("down_resnet: diff=%.6f\n", diff))

            h <- h$transpose(2L, 3L)$contiguous()
            for (i in seq_along(estimator$down_transformers)) {
                h <- estimator$down_transformers[[i]]$forward(h, NULL, t_emb)
            }
            h <- h$transpose(2L, 3L)$contiguous()
            hidden_skip <- h
            h <- estimator$down_conv$forward(h * mask)
            diff <- (h - ref$downsample)$abs()$max()$item()
            cat(sprintf("downsample: diff=%.6f\n", diff))

            for (i in 1:12) {
                h <- estimator$mid_resnets[[i]]$forward(h, mask, t_emb)
                h <- h$transpose(2L, 3L)$contiguous()
                for (j in seq_along(estimator$mid_transformers[[i]])) {
                    h <- estimator$mid_transformers[[i]][[j]]$forward(h, NULL, t_emb)
                }
                h <- h$transpose(2L, 3L)$contiguous()

                ref_name <- paste0("mid_", i - 1, "_after_tfms")
                if (ref_name %in% names(ref)) {
                    diff <- (h - ref[[ref_name]])$abs()$max()$item()
                    cat(sprintf("mid_%d after_tfms: diff=%.6f\n", i - 1, diff))
                }
            }

            diff <- (h - ref$mid_final)$abs()$max()$item()
            cat(sprintf("mid_final: diff=%.6f\n", diff))

            h <- estimator$up_conv$forward(h * mask)
            diff <- (h - ref$upsample)$abs()$max()$item()
            cat(sprintf("upsample: diff=%.6f\n", diff))

            h <- torch::torch_cat(list(h, hidden_skip), dim = 2L)
            h <- estimator$up_resnet$forward(h, mask, t_emb)
            h <- h$transpose(2L, 3L)$contiguous()
            for (i in seq_along(estimator$up_transformers)) {
                h <- estimator$up_transformers[[i]]$forward(h, NULL, t_emb)
            }
            h <- h$transpose(2L, 3L)$contiguous()
            diff <- (h - ref$up_after_tfms)$abs()$max()$item()
            cat(sprintf("up_after_tfms: diff=%.6f\n", diff))

            h <- estimator$final_block$forward(h, mask)
            diff <- (h - ref$final_block)$abs()$max()$item()
            cat(sprintf("final_block: diff=%.6f\n", diff))

            h <- estimator$final_proj$forward(h * mask)
            diff <- (h - ref$final_proj)$abs()$max()$item()
            cat(sprintf("final_proj: diff=%.6f\n", diff))
        }
    })

