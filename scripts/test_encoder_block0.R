#!/usr/bin/env Rscript
# Test encoder block 0 in detail

library(chatterbox)

# Load Python reference
ref <- chatterbox:::read_safetensors("outputs/encoder_steps.safetensors")

# Load weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
weights <- chatterbox:::read_safetensors(weights_path)

# Create encoder
encoder <- chatterbox:::upsample_conformer_encoder_full()
chatterbox:::load_conformer_encoder_weights(encoder, weights, prefix = "flow.encoder.")
encoder$eval()

# Use Python input - get after pre_lookahead
test_input <- ref$input
T <- test_input$size(2)
masks <- torch::torch_ones(c(1L, 1L, T), dtype = torch::torch_bool())

torch::with_no_grad({
        # Run through embed and pre_lookahead to get same starting point
        embed_result <- encoder$embed$forward(test_input, masks)
        xs <- embed_result[[1]]
        pos_emb <- embed_result[[2]]
        xs <- encoder$pre_lookahead_layer$forward(xs)

        cat(sprintf("Input to encoder_0: mean=%.6f, std=%.6f\n", xs$mean()$item(), xs$std()$item()))
        cat(sprintf("Python pre_lookahead: mean=%.6f, std=%.6f\n",
                ref$pre_lookahead$mean()$item(), ref$pre_lookahead$std()$item()))

        # Use Python pre_lookahead output directly
        xs <- ref$pre_lookahead

        # Get encoder block 0
        enc0 <- encoder$encoders[[1]]

        # Step 1: norm_mha
        residual <- xs
        xs_normed <- enc0$norm_mha$forward(xs)
        cat(sprintf("\nAfter norm_mha: mean=%.6f, std=%.6f\n", xs_normed$mean()$item(), xs_normed$std()$item()))

        # Step 2: self_attn
        # Note: R attention signature is (query, key, value, mask, pos_emb)
        attn_out <- enc0$self_attn$forward(xs_normed, xs_normed, xs_normed, masks[, 1,], pos_emb)
        cat(sprintf("After self_attn: mean=%.6f, std=%.6f\n", attn_out$mean()$item(), attn_out$std()$item()))

        # Step 3: dropout + residual
        attn_out <- enc0$dropout$forward(attn_out)
        xs <- residual + attn_out
        cat(sprintf("After attn+residual: mean=%.6f, std=%.6f\n", xs$mean()$item(), xs$std()$item()))

        # Step 4: norm_ff
        residual <- xs
        xs_normed <- enc0$norm_ff$forward(xs)
        cat(sprintf("After norm_ff: mean=%.6f, std=%.6f\n", xs_normed$mean()$item(), xs_normed$std()$item()))

        # Step 5: feed_forward
        ff_out <- enc0$feed_forward$forward(xs_normed)
        cat(sprintf("After feed_forward: mean=%.6f, std=%.6f\n", ff_out$mean()$item(), ff_out$std()$item()))

        # Step 6: dropout + residual
        ff_out <- enc0$dropout$forward(ff_out)
        xs <- residual + ff_out
        cat(sprintf("After ff+residual: mean=%.6f, std=%.6f\n", xs$mean()$item(), xs$std()$item()))

        # Compare
        cat(sprintf("\nPython encoder_0: mean=%.6f, std=%.6f\n",
                ref$encoder_0$mean()$item(), ref$encoder_0$std()$item()))
        diff <- (xs - ref$encoder_0)$abs()$max()$item()
        cat(sprintf("Max diff: %.6f\n", diff))
    })

