#!/usr/bin/env Rscript
# Debug single conformer block step by step

library(torch)
devtools::load_all("/home/troy/chatterbox")

# Load reference - need to save Python block 0 intermediates
# For now, trace through R logic

# Load weights
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")

# Create encoder
encoder <- upsample_conformer_encoder_full()
torch::with_no_grad({
        load_conformer_encoder_weights(encoder, state_dict, prefix = "flow.encoder.")
    })
encoder$eval()

# Load Python reference
py_ref <- safetensors::safe_load_file(
    "/home/troy/chatterbox/outputs/encoder_steps.safetensors",
    framework = "torch"
)

cat("=== Debug conformer block 0 ===\n\n")
torch::with_no_grad({
        # Use Python inputs
        x <- py_ref$after_pll_xs
        pos_emb <- py_ref$after_embed_pos_emb
        masks <- py_ref$masks$to(dtype = torch::torch_bool())

        cat("Input x:", paste(x$shape, collapse = "x"), "mean:", x$mean()$item(), "\n")
        cat("Input pos_emb:", paste(pos_emb$shape, collapse = "x"), "\n")
        cat("Input masks:", paste(masks$shape, collapse = "x"), "\n")

        # Get block
        block <- encoder$encoders[[1]]

        # Step 1: Multi-head attention with pre-norm
        cat("\n--- Step 1: MHA with pre-norm ---\n")
        residual <- x

        # norm_mha
        x_normed <- block$norm_mha$forward(x)
        cat("After norm_mha: mean=", x_normed$mean()$item(), "std=", x_normed$std()$item(), "\n")

        # self_attn
        attn_out <- block$self_attn$forward(x_normed, x_normed, x_normed, masks, pos_emb)
        cat("After self_attn: mean=", attn_out$mean()$item(), "std=", attn_out$std()$item(), "\n")

        # dropout
        attn_out <- block$dropout$forward(attn_out)
        cat("After dropout: mean=", attn_out$mean()$item(), "\n")

        # residual
        x <- residual + attn_out
        cat("After residual: mean=", x$mean()$item(), "std=", x$std()$item(), "\n")

        # Step 2: Feed-forward with pre-norm
        cat("\n--- Step 2: FFN with pre-norm ---\n")
        residual <- x

        # norm_ff
        x_normed <- block$norm_ff$forward(x)
        cat("After norm_ff: mean=", x_normed$mean()$item(), "std=", x_normed$std()$item(), "\n")

        # feed_forward
        ff_out <- block$feed_forward$forward(x_normed)
        cat("After feed_forward: mean=", ff_out$mean()$item(), "std=", ff_out$std()$item(), "\n")

        # dropout
        ff_out <- block$dropout$forward(ff_out)
        cat("After dropout: mean=", ff_out$mean()$item(), "\n")

        # residual
        x <- residual + ff_out
        cat("After residual: mean=", x$mean()$item(), "std=", x$std()$item(), "\n")

        # Step 3: Mask application
        cat("\n--- Step 3: Mask application ---\n")
        mask_pad <- masks
        if (!is.null(mask_pad) && mask_pad$dim() > 0) {
            mask_t <- mask_pad$transpose(2L, 3L)
            cat("mask_t shape:", paste(mask_t$shape, collapse = "x"), "\n")
            x <- x$masked_fill(!mask_t, 0.0)
            cat("After mask: mean=", x$mean()$item(), "std=", x$std()$item(), "\n")
        }

        cat("\n--- Final output ---\n")
        cat("Block output: mean=", x$mean()$item(), "std=", x$std()$item(), "\n")

        # Compare with Python
        # Note: We don't have Python block 0 output saved, only final encoder output
        cat("\n--- Python comparison ---\n")
        cat("Python after all 6 blocks: mean=", py_ref$after_encoders_xs$mean()$item(),
            "std=", py_ref$after_encoders_xs$std()$item(), "\n")
    })

cat("\n=== Done ===\n")

