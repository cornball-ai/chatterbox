#!/usr/bin/env Rscript
# Test Conformer using Python reference inputs at each step

library(torch)
devtools::load_all("/home/troy/chatterbox")

# Load Python reference data
py_ref <- safetensors::safe_load_file(
    "/home/troy/chatterbox/outputs/encoder_steps.safetensors",
    framework = "torch"
)

# Load weights
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")

# Create encoder
encoder <- upsample_conformer_encoder_full()
torch::with_no_grad({
        load_conformer_encoder_weights(encoder, state_dict, prefix = "flow.encoder.")
    })
encoder$eval()

cat("=== Test using Python inputs at each step ===\n\n")
torch::with_no_grad({
        # Step 1: Test pre_lookahead with Python embed output
        cat("=== Step 1: pre_lookahead_layer ===\n")
        py_after_embed <- py_ref$after_embed_xs
        r_pll <- encoder$pre_lookahead_layer$forward(py_after_embed)

        py_pll <- py_ref$after_pll_xs
        pll_diff <- (r_pll - py_pll)$abs()$max()$item()
        cat("Pre-lookahead max diff (using Python embed):", pll_diff, "\n")
        if (pll_diff > 0.01) {
            cat("  R mean:", r_pll$mean()$item(), "std:", r_pll$std()$item(), "\n")
            cat("  Py mean:", py_pll$mean()$item(), "std:", py_pll$std()$item(), "\n")
        }

        # Step 2: Test conformer blocks with Python pll output
        cat("\n=== Step 2: Conformer blocks ===\n")
        py_pos_emb <- py_ref$after_embed_pos_emb
        masks <- py_ref$masks$to(dtype = torch::torch_bool())

        xs_enc <- py_pll# Use Python pll output
        for (i in seq_along(encoder$encoders)) {
            result <- encoder$encoders[[i]]$forward(xs_enc, masks, py_pos_emb, masks)
            xs_enc <- result[[1]]
            cat(sprintf("R after block %d: mean=%.4f, std=%.4f\n",
                    i - 1, xs_enc$mean()$item(), xs_enc$std()$item()))
        }

        py_enc <- py_ref$after_encoders_xs
        enc_diff <- (xs_enc - py_enc)$abs()$max()$item()
        cat("After all blocks max diff:", enc_diff, "\n")

        # Step 3: Test upsample with Python encoder output
        cat("\n=== Step 3: Upsample ===\n")
        py_before_up <- py_ref$before_upsample_xs
        x_lens <- py_ref$input_x_lens$to(dtype = torch::torch_long())

        up_result <- encoder$up_layer$forward(py_before_up, x_lens)
        xs_up <- up_result[[1]]
        xs_up_lens <- up_result[[2]]

        py_up <- py_ref$after_upsample_xs
        up_diff <- (xs_up - py_up)$abs()$max()$item()
        cat("After upsample max diff:", up_diff, "\n")
        cat("R up shape:", paste(xs_up$shape, collapse = "x"), "\n")
        cat("Py up shape:", paste(py_up$shape, collapse = "x"), "\n")

        # Step 4: Test up_embed with Python upsample output
        cat("\n=== Step 4: up_embed ===\n")
        py_up_t <- py_ref$after_upsample_t_xs
        T_up <- py_up_t$size(2)
        xs_up_lens <- torch::torch_tensor(64L) # From Python

        masks_up <- encoder$make_pad_mask(xs_up_lens, T_up, py_up_t$device)$unsqueeze(2)
        masks_up <- !masks_up

        up_embed_result <- encoder$up_embed$forward(py_up_t, masks_up)
        r_xs_up2 <- up_embed_result[[1]]
        r_pos_emb_up <- up_embed_result[[2]]

        py_xs_up2 <- py_ref$after_up_embed_xs
        up_embed_diff <- (r_xs_up2 - py_xs_up2)$abs()$max()$item()
        cat("After up_embed max diff:", up_embed_diff, "\n")

        # Step 5: Test up_encoders with Python up_embed output
        cat("\n=== Step 5: up_encoders ===\n")
        py_pos_emb_up <- py_ref$after_up_embed_pos_emb

        xs_up_enc <- py_xs_up2# Use Python up_embed output
        for (i in seq_along(encoder$up_encoders)) {
            result <- encoder$up_encoders[[i]]$forward(xs_up_enc, masks_up, py_pos_emb_up, masks_up)
            xs_up_enc <- result[[1]]
            cat(sprintf("R after up_block %d: mean=%.4f, std=%.4f\n",
                    i - 1, xs_up_enc$mean()$item(), xs_up_enc$std()$item()))
        }

        py_up_enc <- py_ref$after_up_encoders_xs
        up_enc_diff <- (xs_up_enc - py_up_enc)$abs()$max()$item()
        cat("After up_encoders max diff:", up_enc_diff, "\n")

        # Step 6: Test after_norm with Python up_encoders output
        cat("\n=== Step 6: after_norm ===\n")
        r_final <- encoder$after_norm$forward(py_up_enc)
        py_final <- py_ref$after_norm_xs

        final_diff <- (r_final - py_final)$abs()$max()$item()
        cat("After norm max diff:", final_diff, "\n")

        cat("\n=== Summary ===\n")
        cat("pre_lookahead diff:", pll_diff, "\n")
        cat("conformer blocks diff:", enc_diff, "\n")
        cat("upsample diff:", up_diff, "\n")
        cat("up_embed diff:", up_embed_diff, "\n")
        cat("up_encoders diff:", up_enc_diff, "\n")
        cat("after_norm diff:", final_diff, "\n")

        if (max(pll_diff, enc_diff, up_diff, up_embed_diff, up_enc_diff, final_diff) < 0.01) {
            cat("\nPASS: All components match Python within 0.01\n")
        } else {
            cat("\nSome components have larger differences - need debugging\n")
        }
    })

cat("\n=== Done ===\n")

