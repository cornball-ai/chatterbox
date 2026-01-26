#!/usr/bin/env Rscript
# Debug attention layer step by step

library(torch)
devtools::load_all("/home/troy/chatterbox")

# Load reference
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

cat("=== Debug attention forward ===\n")
torch::with_no_grad({
        # Use Python reference values
        py_xs <- py_ref$after_pll_xs
        py_pos_emb <- py_ref$after_embed_pos_emb
        masks <- py_ref$masks$to(dtype = torch::torch_bool())

        # Get first conformer block's attention
        attn <- encoder$encoders[[1]]$self_attn

        # Full attention forward
        cat("Calling self_attn$forward with:\n")
        cat("  query/key/value:", paste(py_xs$shape, collapse = "x"), "\n")
        cat("  mask:", paste(masks$shape, collapse = "x"), "\n")
        cat("  pos_emb:", paste(py_pos_emb$shape, collapse = "x"), "\n")

        # Step through the forward method manually
        query <- py_xs
        key <- py_xs
        value <- py_xs
        mask <- masks
        pos_emb <- py_pos_emb

        batch_size <- query$size(1)
        seq_len <- query$size(2)
        n_head <- attn$h
        d_k <- attn$d_k

        cat("\nbatch_size:", batch_size, "seq_len:", seq_len, "n_head:", n_head, "d_k:", d_k, "\n")

        # Linear projections
        q <- attn$linear_q$forward(query)$view(c(batch_size, - 1, n_head, d_k))$transpose(2L, 3L)
        k <- attn$linear_k$forward(key)$view(c(batch_size, - 1, n_head, d_k))$transpose(2L, 3L)
        v <- attn$linear_v$forward(value)$view(c(batch_size, - 1, n_head, d_k))$transpose(2L, 3L)
        p <- attn$linear_pos$forward(pos_emb)$view(c(1, - 1, n_head, d_k))$transpose(2L, 3L)

        cat("q:", paste(q$shape, collapse = "x"), "\n")
        cat("k:", paste(k$shape, collapse = "x"), "\n")
        cat("v:", paste(v$shape, collapse = "x"), "\n")
        cat("p:", paste(p$shape, collapse = "x"), "\n")

        # Add positional bias (fixed version)
        q_with_bias_u <- q + attn$pos_bias_u$unsqueeze(1)$unsqueeze(3)
        q_with_bias_v <- q + attn$pos_bias_v$unsqueeze(1)$unsqueeze(3)

        cat("q_with_bias_u:", paste(q_with_bias_u$shape, collapse = "x"), "\n")

        # Content-based attention
        matrix_ac <- torch::torch_matmul(q_with_bias_u, k$transpose(- 2L, - 1L))
        cat("matrix_ac:", paste(matrix_ac$shape, collapse = "x"), "\n")

        # Position-based attention
        matrix_bd <- torch::torch_matmul(q_with_bias_v, p$transpose(- 2L, - 1L))
        cat("matrix_bd before rel_shift:", paste(matrix_bd$shape, collapse = "x"), "\n")

        # rel_shift
        matrix_bd <- attn$rel_shift(matrix_bd)
        cat("matrix_bd after rel_shift:", paste(matrix_bd$shape, collapse = "x"), "\n")

        # Combine and scale
        scores <- (matrix_ac + matrix_bd) / sqrt(d_k)
        cat("scores:", paste(scores$shape, collapse = "x"), "\n")

        # Apply mask
        cat("\nmask:", paste(mask$shape, collapse = "x"), "\n")
        cat("mask dim:", mask$dim(), "\n")

        # Expand mask: (batch, 1, time) -> (batch, 1, 1, time)
        if (mask$dim() == 3) {
            mask_expanded <- mask$unsqueeze(2)
            cat("mask after unsqueeze:", paste(mask_expanded$shape, collapse = "x"), "\n")
        } else {
            mask_expanded <- mask
        }

        # Apply mask
        scores_masked <- scores$masked_fill(!mask_expanded, - 1e9)
        cat("scores_masked:", paste(scores_masked$shape, collapse = "x"), "\n")

        # Softmax
        attn_weights <- torch::nnf_softmax(scores_masked, dim = - 1)
        cat("attn_weights:", paste(attn_weights$shape, collapse = "x"), "\n")

        # Apply to values
        output <- torch::torch_matmul(attn_weights, v)
        cat("output before reshape:", paste(output$shape, collapse = "x"), "\n")

        # Reshape and project
        output <- output$transpose(2L, 3L)$contiguous()$view(c(batch_size, - 1, n_head * d_k))
        cat("output after reshape:", paste(output$shape, collapse = "x"), "\n")

        final_output <- attn$linear_out$forward(output)
        cat("final_output:", paste(final_output$shape, collapse = "x"), "\n")

        cat("\nAttention forward completed successfully.\n")
    })

cat("\n=== Test full conformer block ===\n")
torch::with_no_grad({
        py_xs <- py_ref$after_pll_xs
        py_pos_emb <- py_ref$after_embed_pos_emb
        masks <- py_ref$masks$to(dtype = torch::torch_bool())

        block <- encoder$encoders[[1]]

        cat("Calling conformer block$forward...\n")
        result <- tryCatch({
                block$forward(py_xs, masks, py_pos_emb, masks)
            }, error = function (e)
            {
                cat("Error:", e$message, "\n")
                NULL
            })

        if (!is.null(result)) {
            cat("Block output shape:", paste(result[[1]]$shape, collapse = "x"), "\n")
        }
    })

cat("\n=== Done ===\n")

