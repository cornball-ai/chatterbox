#!/usr/bin/env Rscript
# Trace R estimator intermediate values to compare with Python

library(chatterbox)

cat("Loading references...\n")
euler_ref <- chatterbox:::read_safetensors("outputs/euler_step_trace.safetensors")
py_trace <- chatterbox:::read_safetensors("outputs/estimator_trace.safetensors")

# Load weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

estimator <- chatterbox:::cfm_estimator()
estimator$eval()
loaded <- chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
cat(sprintf("Loaded %d weight keys\n", loaded))

# Get inputs
x_in <- euler_ref$step0_x_in
mu_in <- euler_ref$step0_mu_in
t_in <- euler_ref$step0_t_in
spks_in <- euler_ref$step0_spks_in
cond_in <- euler_ref$step0_cond_in
mask_in <- torch::torch_ones(c(2, 1, 50))

cat("\n=== Step-by-step comparison ===\n")
torch::with_no_grad({
        # 1. Time embedding (sinusoidal)
        t_pos <- estimator$time_embeddings$forward(t_in)$to(dtype = t_in$dtype)
        cat(sprintf("1. time_embeddings: R mean=%.6f (Py: %.6f)\n",
                t_pos$mean()$item(), py_trace$time_pos$mean()$item()))
        diff <- (t_pos - py_trace$time_pos)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # 2. Time MLP
        t_emb <- estimator$time_mlp$forward(t_pos)
        cat(sprintf("2. time_mlp: R mean=%.6f (Py: %.6f)\n",
                t_emb$mean()$item(), py_trace$time_emb$mean()$item()))
        diff <- (t_emb - py_trace$time_emb)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # 3. Input packing
        time <- x_in$size(3) # 50
        h <- torch::torch_cat(list(x_in, mu_in), dim = 2L)
        spks_exp <- spks_in$unsqueeze(3L)$expand(c(- 1L, - 1L, time))
        h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
        h <- torch::torch_cat(list(h, cond_in), dim = 2L)
        cat(sprintf("3. Packed h: R mean=%.6f (Py: %.6f)\n",
                h$mean()$item(), py_trace$packed_h$mean()$item()))
        diff <- (h - py_trace$packed_h)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # 4. Down resnet
        h_res <- estimator$down_resnet$forward(h, mask_in, t_emb)
        cat(sprintf("4. down_resnet: R mean=%.6f (Py: %.6f)\n",
                h_res$mean()$item(), py_trace$down_resnet$mean()$item()))
        diff <- (h_res - py_trace$down_resnet)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # 5. Transformers
        h_attn <- h_res$transpose(2L, 3L)$contiguous()
        attn_mask <- NULL# R uses NULL instead of zeros

        for (i in seq_along(estimator$down_transformers)) {
            h_attn <- estimator$down_transformers[[i]]$forward(h_attn, attn_mask, t_emb)
            cat(sprintf("   transformer[%d]: R mean=%.6f\n", i - 1, h_attn$mean()$item()))
        }

        h_attn <- h_attn$transpose(2L, 3L)$contiguous()
        cat(sprintf("6. after transformers: R mean=%.6f (Py: %.6f)\n",
                h_attn$mean()$item(), py_trace$down_attn$mean()$item()))
        diff <- (h_attn - py_trace$down_attn)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # Skip
        hidden_skip <- h_attn

        # 7. Down conv
        h_down <- estimator$down_conv$forward(h_attn * mask_in)
        cat(sprintf("7. down_conv: R mean=%.6f (Py: %.6f)\n",
                h_down$mean()$item(), py_trace$down_conv$mean()$item()))
        diff <- (h_down - py_trace$down_conv)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # 8. Mid block 0 resnet
        h_mid <- estimator$mid_resnets[[1]]$forward(h_down, mask_in, t_emb)
        cat(sprintf("8. mid[0]_resnet: R mean=%.6f (Py: %.6f)\n",
                h_mid$mean()$item(), py_trace$mid0_resnet$mean()$item()))
        diff <- (h_mid - py_trace$mid0_resnet)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))

        # Full forward
        output <- estimator$forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        cat(sprintf("\nFinal: R mean=%.6f (Py: %.6f)\n",
                output$mean()$item(), py_trace$output$mean()$item()))
        diff <- (output - py_trace$output)$abs()$max()$item()
        cat(sprintf("   diff: %.6f\n", diff))
    })

