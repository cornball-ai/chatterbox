#!/usr/bin/env r
# Test full speech token generation

library(chatterbox)
library(torch)

# Source internal functions
source("~/chatterbox/R/t3.R")

device <- if (cuda_is_available()) "cuda" else "cpu"
cat(sprintf("Using device: %s\n", device))

# Load model
cat("Loading R model...\n")
model <- chatterbox(device)
model <- load_chatterbox(model)
t3 <- model$t3

# Load Python reference inputs
ref <- read_safetensors("~/chatterbox/outputs/llama_layers.safetensors", device = "cpu")
cond_ref <- read_safetensors("~/chatterbox/outputs/cond_enc_debug.safetensors", device = "cpu")

# Create conditioning
speaker_emb <- ref$speaker_emb$to(device = device)
cond_prompt_speech_tokens <- cond_ref$cond_prompt_speech_tokens$to(device = device)
cond <- t3_cond(speaker_emb, cond_prompt_speech_tokens = cond_prompt_speech_tokens, emotion_adv = 0.5)

# Text tokens from Python
text_tokens_cfg <- ref$text_tokens$to(device = device)

# BOS token
bos <- torch::torch_tensor(matrix(t3$config$start_speech_token, nrow = 1), device = device, dtype = torch::torch_long())
bos_cfg <- torch::torch_cat(list(bos, bos), dim = 1L)

# Generation parameters
max_new_tokens <- 200  # Enough for "Hello world" (~1 second)
temperature <- 0.8
cfg_weight <- 0.5
top_p <- 0.95
min_p <- 0.05

cat(sprintf("\nGenerating up to %d speech tokens...\n", max_new_tokens))
cat(sprintf("Parameters: temp=%.1f, cfg=%.1f, top_p=%.2f, min_p=%.2f\n",
            temperature, cfg_weight, top_p, min_p))

# Prepare initial embeddings
prep <- t3$prepare_input_embeds(cond, text_tokens_cfg, bos_cfg, cfg_weight = cfg_weight)
embeds <- prep$embeds

torch::with_no_grad({
  # Initial forward pass
  output <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
  past_key_values <- output$past_key_values

  # Track generated tokens
  generated_ids <- bos[1,, drop = FALSE]$clone()
  predicted <- list()

  # Generation loop
  for (i in seq_len(max_new_tokens)) {
    # Get logits from last position
    logits <- output$last_hidden_state[, output$last_hidden_state$size(2), ]
    logits <- t3$speech_head$forward(logits)

    # CFG combination
    cond_logits <- logits[1, ]$unsqueeze(1)
    uncond_logits <- logits[2, ]$unsqueeze(1)
    logits <- cond_logits + cfg_weight * (cond_logits - uncond_logits)

    # Temperature
    logits <- logits / temperature

    # Convert to probs for sampling
    probs <- torch::nnf_softmax(logits, dim = -1L)

    # Min-p filtering
    max_prob <- probs$max()
    min_threshold <- min_p * max_prob
    logits[probs < min_threshold] <- -Inf

    # Recompute probs after min-p filtering (CRITICAL: Python does this)
    probs_filtered <- torch::nnf_softmax(logits, dim = -1L)

    # Top-p sampling
    sorted_result <- torch::torch_sort(probs_filtered, descending = TRUE)
    sorted_probs <- sorted_result[[1]]
    sorted_indices <- sorted_result[[2]]
    cumsum_probs <- torch::torch_cumsum(sorted_probs, dim = -1L)

    sorted_mask <- cumsum_probs > top_p
    sorted_mask[, 1] <- FALSE
    sorted_probs[sorted_mask] <- 0

    # Re-normalize
    sorted_probs <- sorted_probs / sorted_probs$sum()

    # Sample
    next_token_idx <- torch::torch_multinomial(sorted_probs, num_samples = 1L)
    next_token <- sorted_indices$gather(2L, next_token_idx)

    # Note: sorted_indices returns R 1-indexed positions
    # Convert to 0-indexed token ID by subtracting 1
    token_id <- as.integer(next_token$item()) - 1L
    predicted[[length(predicted) + 1]] <- next_token
    generated_ids <- torch::torch_cat(list(generated_ids, next_token), dim = 2L)

    # Check for EOS (comparing 0-indexed token IDs)
    if (token_id == t3$config$stop_speech_token) {
      cat(sprintf("EOS detected at step %d\n", i))
      break
    }

    # Progress every 20 tokens
    if (i %% 20 == 0) {
      # stop_speech_token is 0-indexed, need +1 for R tensor indexing
      eos_prob <- probs[1, t3$config$stop_speech_token + 1L]$item()
      cat(sprintf("Step %d: token=%d (0-indexed), EOS_prob=%.6f\n", i, token_id, eos_prob))
    }

    # Get embedding for next token
    # sorted_indices returns R 1-indexed values, which nn_embedding expects
    next_emb <- t3$speech_emb$forward(next_token) + t3$speech_pos_emb$get_fixed_embedding(i)

    # Double for CFG
    next_emb <- torch::torch_cat(list(next_emb, next_emb), dim = 1L)

    # Forward with KV cache
    output <- t3$tfmr$forward(inputs_embeds = next_emb, past_key_values = past_key_values, use_cache = TRUE)
    past_key_values <- output$past_key_values
  }
})

# Summary
n_tokens <- length(predicted)
cat(sprintf("\nGenerated %d speech tokens\n", n_tokens))

# Approximate audio duration (assuming 86 tokens/second based on codec)
tokens_per_second <- 86
duration_sec <- n_tokens / tokens_per_second
cat(sprintf("Approximate audio duration: %.2f seconds\n", duration_sec))

if (n_tokens > 0) {
  tokens_vec <- sapply(predicted, function(t) as.integer(t$item()))
  cat(sprintf("First 10 tokens: %s\n", paste(head(tokens_vec, 10), collapse = ", ")))
  cat(sprintf("Last 10 tokens: %s\n", paste(tail(tokens_vec, 10), collapse = ", ")))
}

cat("\nDone.\n")
