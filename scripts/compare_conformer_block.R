#!/usr/bin/env Rscript
# Compare R vs Python conformer block step by step

library(torch)
devtools::load_all("/home/troy/chatterbox")

# Load Python block intermediates
py_block <- safetensors::safe_load_file(
    "/home/troy/chatterbox/outputs/conformer_block0_steps.safetensors",
    framework = "torch"
)

cat("Python intermediates:\n")
for (k in names(py_block)) {
    cat(sprintf("  %s: %s\n", k, paste(py_block[[k]]$shape, collapse = "x")))
}

# Load weights
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")

# Create encoder
encoder <- upsample_conformer_encoder_full()
torch::with_no_grad({
        load_conformer_encoder_weights(encoder, state_dict, prefix = "flow.encoder.")
    })
encoder$eval()

block <- encoder$encoders[[1]]

cat("\n=== Compare each step ===\n")
torch::with_no_grad({
        # Inputs from Python
        x <- py_block$input_x
        pos_emb <- py_block$input_pos_emb
        masks <- py_block$input_masks$to(dtype = torch::torch_bool())

        # Step 1: norm_mha
        cat("\n--- norm_mha ---\n")
        r_norm_mha <- block$norm_mha$forward(x)
        py_norm_mha <- py_block$after_norm_mha

        diff <- (r_norm_mha - py_norm_mha)$abs()$max()$item()
        cat("norm_mha max diff:", diff, "\n")
        cat("R  mean:", r_norm_mha$mean()$item(), "std:", r_norm_mha$std()$item(), "\n")
        cat("Py mean:", py_norm_mha$mean()$item(), "std:", py_norm_mha$std()$item(), "\n")

        # Step 2: self_attn (using Python norm_mha output to isolate attention)
        cat("\n--- self_attn (using Python norm_mha) ---\n")
        r_attn <- block$self_attn$forward(py_norm_mha, py_norm_mha, py_norm_mha, masks, pos_emb)
        py_attn <- py_block$after_self_attn

        diff <- (r_attn - py_attn)$abs()$max()$item()
        cat("self_attn max diff:", diff, "\n")
        cat("R  mean:", r_attn$mean()$item(), "std:", r_attn$std()$item(), "\n")
        cat("Py mean:", py_attn$mean()$item(), "std:", py_attn$std()$item(), "\n")

        if (diff > 0.01) {
            cat("\n  Attention difference too large - need to debug attention mechanism\n")
        }

        # Step 3: MHA residual
        cat("\n--- MHA residual ---\n")
        residual <- x
        r_mha_res <- residual + r_attn
        py_mha_res <- py_block$after_mha_residual

        diff <- (r_mha_res - py_mha_res)$abs()$max()$item()
        cat("MHA residual max diff:", diff, "\n")

        # Step 4: norm_ff (using Python MHA output)
        cat("\n--- norm_ff (using Python MHA output) ---\n")
        r_norm_ff <- block$norm_ff$forward(py_mha_res)
        py_norm_ff <- py_block$after_norm_ff

        diff <- (r_norm_ff - py_norm_ff)$abs()$max()$item()
        cat("norm_ff max diff:", diff, "\n")

        # Step 5: feed_forward (using Python norm_ff output)
        cat("\n--- feed_forward (using Python norm_ff output) ---\n")
        r_ff <- block$feed_forward$forward(py_norm_ff)
        py_ff <- py_block$after_ff

        diff <- (r_ff - py_ff)$abs()$max()$item()
        cat("feed_forward max diff:", diff, "\n")
        cat("R  mean:", r_ff$mean()$item(), "std:", r_ff$std()$item(), "\n")
        cat("Py mean:", py_ff$mean()$item(), "std:", py_ff$std()$item(), "\n")

        # Step 6: Full block output (using Python FF output)
        cat("\n--- Final residual ---\n")
        r_final <- py_mha_res + r_ff
        py_final <- py_block$block0_full_output

        diff <- (r_final - py_final)$abs()$max()$item()
        cat("Final output max diff:", diff, "\n")
    })

cat("\n=== Done ===\n")

