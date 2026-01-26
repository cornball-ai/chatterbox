#!/usr/bin/env Rscript
# Test encoder step by step against Python reference

library(chatterbox)

# Load Python reference
cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/encoder_steps.safetensors")

# Load weights
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
weights <- chatterbox:::read_safetensors(weights_path)

# Create encoder
encoder <- chatterbox:::upsample_conformer_encoder_full(
    input_size = 512L,
    output_size = 512L,
    num_blocks = 6L,
    num_up_blocks = 4L
)
chatterbox:::load_conformer_encoder_weights(encoder, weights, prefix = "flow.encoder.")
encoder$eval()

# Use Python input
test_input <- ref$input
test_lens <- torch::torch_tensor(50L)$view(c(1L))

cat("\nPython intermediate values:\n")
cat(sprintf("  input: mean=%.6f, std=%.6f\n", ref$input$mean()$item(), ref$input$std()$item()))
cat(sprintf("  embed: mean=%.6f, std=%.6f\n", ref$embed$mean()$item(), ref$embed$std()$item()))
cat(sprintf("  pre_lookahead: mean=%.6f, std=%.6f\n", ref$pre_lookahead$mean()$item(), ref$pre_lookahead$std()$item()))
cat(sprintf("  encoder_0: mean=%.6f, std=%.6f\n", ref$encoder_0$mean()$item(), ref$encoder_0$std()$item()))
cat(sprintf("  encoder_5: mean=%.6f, std=%.6f\n", ref$encoder_5$mean()$item(), ref$encoder_5$std()$item()))
cat(sprintf("  output: mean=%.6f, std=%.6f\n", ref$output$mean()$item(), ref$output$std()$item()))

cat("\n\nR step-by-step forward:\n")

torch::with_no_grad({
        # Create masks
        T <- test_input$size(2)
        masks <- torch::torch_ones(c(1L, 1L, T), dtype = torch::torch_bool())

        # 1. Embed
        embed_result <- encoder$embed$forward(test_input, masks)
        xs <- embed_result[[1]]
        pos_emb <- embed_result[[2]]
        cat(sprintf("  embed: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(xs), collapse = "x"), xs$mean()$item(), xs$std()$item()))
        embed_diff <- (xs - ref$embed)$abs()$max()$item()
        cat(sprintf("    diff vs Python: %.6f\n", embed_diff))

        # 2. Pre-lookahead
        xs <- encoder$pre_lookahead_layer$forward(xs)
        cat(sprintf("  pre_lookahead: mean=%.6f, std=%.6f\n", xs$mean()$item(), xs$std()$item()))
        pre_diff <- (xs - ref$pre_lookahead)$abs()$max()$item()
        cat(sprintf("    diff vs Python: %.6f\n", pre_diff))

        # 3. Encoder blocks (note: forward args are x, mask, pos_emb)
        for (i in seq_along(encoder$encoders)) {
            result <- encoder$encoders[[i]]$forward(xs, masks[, 1,], pos_emb)
            xs <- result[[1]]# Forward returns list(x, mask, ...)
            ref_name <- paste0("encoder_", i - 1)
            ref_tensor <- ref[[ref_name]]
            cat(sprintf("  encoder_%d: mean=%.6f, std=%.6f\n", i - 1, xs$mean()$item(), xs$std()$item()))
            diff <- (xs - ref_tensor)$abs()$max()$item()
            cat(sprintf("    diff vs Python: %.6f\n", diff))
            if (diff > 0.1) {
                cat("    *** LARGE DIFFERENCE ***\n")
            }
        }

        # 4. Up layer (expects (batch, channels, time))
        # Transpose: (B, T, D) -> (B, D, T)
        xs_t <- xs$transpose(2L, 3L)$contiguous()
        xs_up <- encoder$up_layer$forward(xs_t, torch::torch_tensor(50L)$view(c(1L)))
        xs_up <- xs_up[[1]]# Returns (tensor, lens)
        # Transpose back: (B, D, T) -> (B, T, D)
        xs_up <- xs_up$transpose(2L, 3L)$contiguous()
        cat(sprintf("  up_layer: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(xs_up), collapse = "x"), xs_up$mean()$item(), xs_up$std()$item()))

        # 5. Up embed
        masks_up <- torch::torch_ones(c(1L, 1L, xs_up$size(2)), dtype = torch::torch_bool())
        up_embed_result <- encoder$up_embed$forward(xs_up, masks_up)
        xs_up <- up_embed_result[[1]]
        pos_emb_up <- up_embed_result[[2]]
        cat(sprintf("  up_embed: mean=%.6f, std=%.6f\n", xs_up$mean()$item(), xs_up$std()$item()))

        # 6. Up encoders
        for (i in seq_along(encoder$up_encoders)) {
            result_up <- encoder$up_encoders[[i]]$forward(xs_up, masks_up[, 1,], pos_emb_up)
            xs_up <- result_up[[1]]
            ref_name <- paste0("up_encoder_", i - 1)
            ref_tensor <- ref[[ref_name]]
            cat(sprintf("  up_encoder_%d: mean=%.6f, std=%.6f\n", i - 1, xs_up$mean()$item(), xs_up$std()$item()))
            diff <- (xs_up - ref_tensor)$abs()$max()$item()
            cat(sprintf("    diff vs Python: %.6f\n", diff))
        }

        # 7. After norm
        xs_final <- encoder$after_norm$forward(xs_up)
        cat(sprintf("  after_norm: mean=%.6f, std=%.6f\n", xs_final$mean()$item(), xs_final$std()$item()))
        final_diff <- (xs_final - ref$after_norm)$abs()$max()$item()
        cat(sprintf("    diff vs Python: %.6f\n", final_diff))

        # Compare to full output
        output_diff <- (xs_final - ref$output)$abs()$max()$item()
        cat(sprintf("\nFinal output diff: %.6f\n", output_diff))
    })

