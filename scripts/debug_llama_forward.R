#!/usr/bin/env Rscript
# Debug Llama forward pass step by step

rhydrogen::load_all()

cat("=== Loading Model ===\n")
device <- "cuda"
model <- chatterbox(device)
model <- load_chatterbox(model)

llama <- model$t3$tfmr

# Create simple input
cat("\n=== Test Forward Pass Step by Step ===\n")
test_input <- torch::torch_randn(c(1, 5, 1024), device = device)
cat("Input shape:", paste(dim(test_input), collapse = "x"), "\n")
cat("Input mean:", test_input$mean()$item(), "\n")
cat("Input std:", test_input$std()$item(), "\n")

torch::with_no_grad({
        # Step 1: Hidden states initialization
        hidden_states <- test_input
        cat("\nStep 1 - Initial hidden states:\n")
        cat("  shape:", paste(dim(hidden_states), collapse = "x"), "\n")
        cat("  mean:", hidden_states$mean()$item(), "\n")
        cat("  range: [", hidden_states$min()$item(), ", ", hidden_states$max()$item(), "]\n", sep = "")

        # Step 2: Get position IDs
        batch_size <- hidden_states$size(1)
        seq_length <- hidden_states$size(2)
        position_ids <- torch::torch_arange(0, seq_length - 1, device = device, dtype = torch::torch_long())
        position_ids <- position_ids$unsqueeze(1)$expand(c(batch_size, - 1))
        cat("\nStep 2 - Position IDs:\n")
        cat("  shape:", paste(dim(position_ids), collapse = "x"), "\n")
        cat("  values:", paste(as.integer(position_ids$cpu()), collapse = ", "), "\n")

        # Step 3: Get RoPE cache
        rope_cache <- llama$.get_rope_cache(seq_length, device)
        cat("\nStep 3 - RoPE cache:\n")
        cat("  cos shape:", paste(dim(rope_cache$cos), collapse = "x"), "\n")
        cat("  sin shape:", paste(dim(rope_cache$sin), collapse = "x"), "\n")
        cat("  cos mean:", rope_cache$cos$mean()$item(), "\n")

        # Step 4: Create causal mask
        mask <- torch::torch_full(c(seq_length, seq_length), - Inf, device = device)
        mask <- torch::torch_triu(mask, diagonal = 1)
        attention_mask <- mask$unsqueeze(1)$unsqueeze(1)
        cat("\nStep 4 - Causal mask:\n")
        cat("  shape:", paste(dim(attention_mask), collapse = "x"), "\n")

        # Step 5: Process first layer manually
        layer <- llama$layers[[1]]
        cat("\nStep 5 - First Layer:\n")

        # Pre-norm
        residual <- hidden_states
        normed <- layer$input_layernorm$forward(hidden_states)
        cat("  After input_layernorm - mean:", normed$mean()$item(), ", std:", normed$std()$item(), "\n")

        # Self attention
        bsz <- normed$size(1)
        q_len <- normed$size(2)

        query_states <- layer$self_attn$q_proj$forward(normed)
        key_states <- layer$self_attn$k_proj$forward(normed)
        value_states <- layer$self_attn$v_proj$forward(normed)

        cat("  query_states - mean:", query_states$mean()$item(), ", std:", query_states$std()$item(), "\n")
        cat("  key_states - mean:", key_states$mean()$item(), ", std:", key_states$std()$item(), "\n")
        cat("  value_states - mean:", value_states$mean()$item(), ", std:", value_states$std()$item(), "\n")

        # Reshape
        num_heads <- llama$config$num_attention_heads
        head_dim <- llama$config$head_dim
        query_states <- query_states$view(c(bsz, q_len, num_heads, head_dim))$transpose(2, 3)
        key_states <- key_states$view(c(bsz, q_len, num_heads, head_dim))$transpose(2, 3)
        value_states <- value_states$view(c(bsz, q_len, num_heads, head_dim))$transpose(2, 3)

        cat("  Reshaped query - shape:", paste(dim(query_states), collapse = "x"), "\n")

        # Apply RoPE
        rotated <- apply_rotary_pos_emb(query_states, key_states, rope_cache$cos, rope_cache$sin, position_ids)
        query_states <- rotated$q
        key_states <- rotated$k

        cat("  After RoPE - query mean:", query_states$mean()$item(), "\n")

        # Attention
        scale <- 1.0 / sqrt(head_dim)
        attn_weights <- torch::torch_matmul(query_states, key_states$transpose(3, 4)) * scale
        cat("  Attention weights before mask - mean:", attn_weights$mean()$item(), "\n")

        attn_weights <- attn_weights + attention_mask
        attn_weights <- torch::nnf_softmax(attn_weights, dim = - 1)
        cat("  Attention weights after softmax - mean:", attn_weights$mean()$item(), "\n")

        attn_output <- torch::torch_matmul(attn_weights, value_states)
        cat("  Attention output - mean:", attn_output$mean()$item(), "\n")

        # Reshape back
        attn_output <- attn_output$transpose(2, 3)$contiguous()
        attn_output <- attn_output$view(c(bsz, q_len, num_heads * head_dim))

        # Output projection
        attn_output <- layer$self_attn$o_proj$forward(attn_output)
        cat("  After o_proj - mean:", attn_output$mean()$item(), "\n")

        # Add residual
        hidden_states_after_attn <- residual + attn_output
        cat("  After attention residual - mean:", hidden_states_after_attn$mean()$item(), "\n")

        # MLP
        residual <- hidden_states_after_attn
        mlp_input <- layer$post_attention_layernorm$forward(hidden_states_after_attn)
        cat("  After post_attention_layernorm - mean:", mlp_input$mean()$item(), "\n")

        gate <- layer$mlp$gate_proj$forward(mlp_input)
        up <- layer$mlp$up_proj$forward(mlp_input)
        cat("  Gate - mean:", gate$mean()$item(), "\n")
        cat("  Up - mean:", up$mean()$item(), "\n")

        # SiLU
        silu_gate <- gate * torch::torch_sigmoid(gate)
        cat("  After SiLU - mean:", silu_gate$mean()$item(), "\n")

        mlp_output <- layer$mlp$down_proj$forward(silu_gate * up)
        cat("  After down_proj - mean:", mlp_output$mean()$item(), "\n")

        hidden_states_after_mlp <- residual + mlp_output
        cat("  After MLP residual - mean:", hidden_states_after_mlp$mean()$item(), "\n")

        # Step 6: Final norm
        cat("\nStep 6 - Final norm:\n")
        cat("  norm weight mean:", llama$norm$weight$mean()$item(), "\n")
        final_output <- llama$norm$forward(hidden_states_after_mlp)
        cat("  Final output - mean:", final_output$mean()$item(), "\n")
        cat("  Final output - std:", final_output$std()$item(), "\n")
    })

