#!/usr/bin/env Rscript
# Check attention mechanism in CFM transformer

library(chatterbox)

cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

# Load weights for a single transformer block
estimator <- chatterbox:::cfm_estimator()
chatterbox:::load_cfm_estimator_weights(estimator, state_dict, prefix = "flow.decoder.estimator.")
estimator$eval()

# Get first mid block transformer
tfm <- estimator$mid_transformers[[1]][[1]]

cat("=== Transformer block structure ===\n")
cat(sprintf("norm1 weight shape: %s\n", paste(dim(tfm$norm1$weight), collapse = "x")))
cat(sprintf("attn1.to_q weight shape: %s\n", paste(dim(tfm$attn1$to_q$weight), collapse = "x")))
cat(sprintf("attn1.to_out[[1]] weight shape: %s\n", paste(dim(tfm$attn1$to_out[[1]]$weight), collapse = "x")))

# Create test input
torch::with_no_grad({
        # Simple test input
        hidden <- torch::torch_randn(c(1L, 50L, 256L)) * 0.1# (B, T, C) small values
        t_emb <- torch::torch_randn(c(1L, 1024L))

        cat("\n=== Testing attention forward ===\n")
        cat(sprintf("Input: mean=%.4f, std=%.4f\n", hidden$mean()$item(), hidden$std()$item()))

        # Norm
        norm_hidden <- tfm$norm1$forward(hidden)
        cat(sprintf("After norm1: mean=%.4f, std=%.4f\n", norm_hidden$mean()$item(), norm_hidden$std()$item()))

        # Attention
        attn <- tfm$attn1
        batch_size <- norm_hidden$size(1)
        seq_len <- norm_hidden$size(2)

        q <- attn$to_q$forward(norm_hidden)
        k <- attn$to_k$forward(norm_hidden)
        v <- attn$to_v$forward(norm_hidden)

        cat(sprintf("q: mean=%.4f, std=%.4f\n", q$mean()$item(), q$std()$item()))
        cat(sprintf("k: mean=%.4f, std=%.4f\n", k$mean()$item(), k$std()$item()))
        cat(sprintf("v: mean=%.4f, std=%.4f\n", v$mean()$item(), v$std()$item()))

        # Reshape for multi-head
        q <- q$view(c(batch_size, seq_len, attn$heads, attn$head_dim))$transpose(2L, 3L)
        k <- k$view(c(batch_size, seq_len, attn$heads, attn$head_dim))$transpose(2L, 3L)
        v <- v$view(c(batch_size, seq_len, attn$heads, attn$head_dim))$transpose(2L, 3L)

        cat(sprintf("q reshaped: %s\n", paste(dim(q), collapse = "x")))

        # Attention scores
        scores <- torch::torch_matmul(q, k$transpose(- 2L, - 1L)) * attn$scale
        cat(sprintf("scores: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
                scores$mean()$item(), scores$std()$item(),
                scores$min()$item(), scores$max()$item()))

        attn_probs <- torch::nnf_softmax(scores, dim = - 1L)
        cat(sprintf("attn_probs: mean=%.4f\n", attn_probs$mean()$item()))

        out <- torch::torch_matmul(attn_probs, v)
        out <- out$transpose(2L, 3L)$contiguous()$view(c(batch_size, seq_len, - 1L))
        cat(sprintf("attention output (before proj): mean=%.4f, std=%.4f\n", out$mean()$item(), out$std()$item()))

        out <- attn$to_out[[1]]$forward(out)
        cat(sprintf("attention output (after proj): mean=%.4f, std=%.4f\n", out$mean()$item(), out$std()$item()))

        # Full transformer forward
        cat("\n=== Full transformer forward ===\n")
        out_full <- tfm$forward(hidden, NULL, t_emb)
        cat(sprintf("Transformer output: mean=%.4f, std=%.4f\n", out_full$mean()$item(), out_full$std()$item()))
    })

# Check if Python attention uses different scale
cat("\n=== Scale check ===\n")
cat(sprintf("head_dim: %d\n", tfm$attn1$head_dim))
cat(sprintf("scale: %.6f\n", tfm$attn1$scale))
cat(sprintf("sqrt(head_dim): %.6f\n", sqrt(tfm$attn1$head_dim)))

