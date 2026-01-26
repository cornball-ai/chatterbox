#!/usr/bin/env Rscript
# Debug Conformer Encoder step by step

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

# Test input
x <- py_ref$input_x
x_lens <- py_ref$input_x_lens$to(dtype = torch::torch_long())

cat("=== Debug embed layer ===\n")
torch::with_no_grad({
        # Check weight values
        w1 <- encoder$embed$out[[1]]$weight
        py_w1 <- state_dict[["flow.encoder.embed.out.0.weight"]]
        cat("Linear weight diff:", (w1 - py_w1)$abs()$max()$item(), "\n")

        # Run through linear part only (no pos enc)
        T <- x$size(2)
        device <- x$device
        masks <- encoder$make_pad_mask(x_lens, T, device)$unsqueeze(2)
        masks <- !masks

        # Just the sequential part
        xs_linear <- encoder$embed$out$forward(x)
        cat("After linear xs shape:", paste(xs_linear$shape, collapse = "x"), "\n")
        cat("After linear xs mean:", xs_linear$mean()$item(), "\n")

        # Python after embed
        py_xs <- py_ref$after_embed_xs
        cat("Python xs mean:", py_xs$mean()$item(), "\n")

        # Compare without pos encoding
        linear_diff <- (xs_linear - py_xs)$abs()$max()$item()
        cat("Linear part max diff:", linear_diff, "\n")
    })

cat("\n=== Debug attention shapes ===\n")
torch::with_no_grad({
        # Use Python reference values for debugging
        py_xs <- py_ref$after_pll_xs
        py_pos_emb <- py_ref$after_embed_pos_emb
        masks <- py_ref$masks$to(dtype = torch::torch_bool())

        cat("py_xs shape:", paste(py_xs$shape, collapse = "x"), "\n")
        cat("py_pos_emb shape:", paste(py_pos_emb$shape, collapse = "x"), "\n")
        cat("masks shape:", paste(masks$shape, collapse = "x"), "\n")

        # Get first conformer block
        block0 <- encoder$encoders[[1]]

        # Run attention layer step by step
        attn <- block0$self_attn

        # Input to attention
        query <- py_xs
        key <- py_xs
        value <- py_xs
        pos_emb <- py_pos_emb

        batch_size <- query$size(1)
        seq_len <- query$size(2)
        cat("\nbatch_size:", batch_size, "seq_len:", seq_len, "\n")

        # Linear projections
        q <- attn$linear_q$forward(query)
        cat("q shape after linear_q:", paste(q$shape, collapse = "x"), "\n")

        # Reshape to (batch, head, time, d_k)
        n_head <- attn$h
        d_k <- attn$d_k
        cat("n_head:", n_head, "d_k:", d_k, "\n")

        q <- q$view(c(batch_size, - 1, n_head, d_k))$transpose(2L, 3L)
        cat("q after reshape:", paste(q$shape, collapse = "x"), "\n")

        k <- attn$linear_k$forward(key)$view(c(batch_size, - 1, n_head, d_k))$transpose(2L, 3L)
        v <- attn$linear_v$forward(value)$view(c(batch_size, - 1, n_head, d_k))$transpose(2L, 3L)
        cat("k shape:", paste(k$shape, collapse = "x"), "\n")
        cat("v shape:", paste(v$shape, collapse = "x"), "\n")

        # Project pos encoding
        p <- attn$linear_pos$forward(pos_emb)
        cat("p after linear_pos:", paste(p$shape, collapse = "x"), "\n")

        p <- p$view(c(1, - 1, n_head, d_k))$transpose(2L, 3L)
        cat("p after reshape:", paste(p$shape, collapse = "x"), "\n")

        # Add positional bias to query
        pos_bias_u <- attn$pos_bias_u
        pos_bias_v <- attn$pos_bias_v
        cat("pos_bias_u shape:", paste(pos_bias_u$shape, collapse = "x"), "\n")

        # Expand pos_bias_u: (n_head, d_k) -> (1, n_head, 1, d_k)
        # pos_bias_u is (8, 64), we need (1, 8, 1, 64) to broadcast with q (1, 8, 32, 64)
        # unsqueeze(1) adds dim at beginning: (8, 64) -> (1, 8, 64)
        # unsqueeze(3) adds dim at pos 2: (1, 8, 64) -> (1, 8, 1, 64)
        bias_u <- pos_bias_u$unsqueeze(1)$unsqueeze(3)
        cat("bias_u after unsqueeze:", paste(bias_u$shape, collapse = "x"), "\n")

        q_with_bias_u <- q + bias_u
        cat("q_with_bias_u shape:", paste(q_with_bias_u$shape, collapse = "x"), "\n")

        # Content-based attention
        k_t <- k$transpose(- 2L, - 1L)
        cat("k transposed:", paste(k_t$shape, collapse = "x"), "\n")

        matrix_ac <- torch::torch_matmul(q_with_bias_u, k_t)
        cat("matrix_ac shape:", paste(matrix_ac$shape, collapse = "x"), "\n")

        # Position-based attention
        q_with_bias_v <- q + pos_bias_v$unsqueeze(1)$unsqueeze(3)
        p_t <- p$transpose(- 2L, - 1L)
        cat("p transposed:", paste(p_t$shape, collapse = "x"), "\n")

        matrix_bd <- torch::torch_matmul(q_with_bias_v, p_t)
        cat("matrix_bd shape:", paste(matrix_bd$shape, collapse = "x"), "\n")

        # The rel_shift expects (batch, head, time, 2*time-1)
        # Let's check what we have
        cat("\nExpected pos_len: 2 * seq_len - 1 =", 2 * seq_len - 1, "\n")
        cat("Actual pos_len:", matrix_bd$size(4), "\n")
    })

cat("\n=== Done ===\n")

