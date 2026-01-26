#!/usr/bin/env Rscript
# Detailed CFM estimator debugging

library(chatterbox)

cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/cfm_estimator_steps.safetensors")

# Load s3gen weights
cat("Loading s3gen weights...\n")
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

# Create estimator
estimator <- chatterbox:::cfm_estimator()
estimator$eval()

# Load weights and check warnings
cat("\nLoading weights...\n")
loaded <- chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
cat(sprintf("Loaded %d weight keys\n", loaded))

# Print warnings
w <- warnings()
if (length(w) > 0) {
    cat("\nWarnings:\n")
    print(w)
}

# Get inputs
x <- ref$input_x
mask <- ref$input_mask
mu <- ref$input_mu
t <- ref$input_t
spks <- ref$input_spks
cond <- ref$input_cond

cat("\n=== Step-by-step validation ===\n")
torch::with_no_grad({
        # 1. Time embedding
        t_emb_sin <- estimator$time_embeddings$forward(t)
        t_emb <- estimator$time_mlp$forward(t_emb_sin)
        cat(sprintf("time_emb: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                t_emb$mean()$item(), ref$time_emb_mlp$mean()$item(),
                (t_emb - ref$time_emb_mlp)$abs()$max()$item()))

        # 2. Pack inputs
        batch_size <- x$size(1)
        seq_len <- x$size(3)
        h <- torch::torch_cat(list(x, mu), dim = 2L)
        spks_exp <- spks$unsqueeze(3L)$expand(c(- 1L, - 1L, seq_len))
        h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
        h <- torch::torch_cat(list(h, cond), dim = 2L)

        cat(sprintf("packed_all: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                h$mean()$item(), ref$packed_all$mean()$item(),
                (h - ref$packed_all)$abs()$max()$item()))

        # 3. Down resnet
        h <- estimator$down_resnet$forward(h, mask, t_emb)
        cat(sprintf("down_resnet: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                h$mean()$item(), ref$down_resnet$mean()$item(),
                (h - ref$down_resnet)$abs()$max()$item()))

        # 4. Down transformers
        h <- h$transpose(2L, 3L)$contiguous() # (B, T, C)
        attn_mask <- NULL# No masking for now

        h <- estimator$down_transformers[[1]]$forward(h, attn_mask, t_emb)
        # Python down_tfm_0 is (B, T, C) after transformer
        cat(sprintf("down_tfm_0: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                h$mean()$item(), ref$down_tfm_0$mean()$item(),
                (h - ref$down_tfm_0)$abs()$max()$item()))

        # Continue transformers
        for (i in 2:4) {
            h <- estimator$down_transformers[[i]]$forward(h, attn_mask, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous() # (B, C, T)

        hidden_skip <- h

        # 5. Downsample
        h <- estimator$down_conv$forward(h * mask)
        cat(sprintf("downsample: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                h$mean()$item(), ref$downsample$mean()$item(),
                (h - ref$downsample)$abs()$max()$item()))

        # 6. Mid block 0
        h <- estimator$mid_resnets[[1]]$forward(h, mask, t_emb)
        cat(sprintf("mid_0_resnet: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                h$mean()$item(), ref$mid_0_resnet$mean()$item(),
                (h - ref$mid_0_resnet)$abs()$max()$item()))

        h <- h$transpose(2L, 3L)$contiguous()
        h <- estimator$mid_transformers[[1]][[1]]$forward(h, attn_mask, t_emb)
        cat(sprintf("mid_0_tfm_0: R mean=%.6f, Py mean=%.6f, diff=%.6f\n",
                h$mean()$item(), ref$mid_0_tfm_0$mean()$item(),
                (h - ref$mid_0_tfm_0)$abs()$max()$item()))
    })

