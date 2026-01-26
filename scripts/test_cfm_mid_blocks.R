#!/usr/bin/env Rscript
# Trace through mid blocks individually

library(chatterbox)

cat("Loading references...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")

cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

estimator <- chatterbox:::cfm_estimator()
estimator$eval()
loaded <- chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")

# Get inputs
x <- ref$input_x
mask <- ref$input_mask
mu <- ref$input_mu
t <- ref$input_t
spks <- ref$input_spks
cond <- ref$input_cond

cat("\n=== Mid block trace ===\n")
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
        for (i in seq_along(estimator$down_transformers)) {
            h <- estimator$down_transformers[[i]]$forward(h, NULL, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous()
        hidden_skip <- h
        h <- estimator$down_conv$forward(h * mask)

        # Downsample matches
        diff <- (h - ref$downsample)$abs()$max()$item()
        cat(sprintf("After downsample: diff=%.6f\n", diff))

        # Track each mid block
        for (i in 1:12) {
            # Resnet
            h <- estimator$mid_resnets[[i]]$forward(h, mask, t_emb)

            ref_resnet_name <- paste0("mid_", i - 1, "_resnet")
            if (ref_resnet_name %in% names(ref)) {
                ref_res <- ref[[ref_resnet_name]]
                diff <- (h - ref_res)$abs()$max()$item()
                cat(sprintf("mid_%d resnet: R mean=%.4f, Py mean=%.4f, diff=%.6f\n",
                        i - 1, h$mean()$item(), ref_res$mean()$item(), diff))
            }

            # Transformers
            h <- h$transpose(2L, 3L)$contiguous()
            for (j in seq_along(estimator$mid_transformers[[i]])) {
                h <- estimator$mid_transformers[[i]][[j]]$forward(h, NULL, t_emb)
            }
            h <- h$transpose(2L, 3L)$contiguous()
        }

        cat(sprintf("\nFinal after all mid blocks: mean=%.4f, std=%.4f\n",
                h$mean()$item(), h$std()$item()))
    })

