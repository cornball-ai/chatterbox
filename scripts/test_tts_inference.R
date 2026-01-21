#!/usr/bin/env r
# Test full TTS inference and compare with Python

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

# Test text
text <- "Hello world"
cat(sprintf("Text: %s\n", text))

# Prepare conditioning from reference audio
ref_audio <- "~/chatterbox/inst/audio/jfk.wav"
cat(sprintf("Reference audio: %s\n", ref_audio))

# Load from Python reference (already tested that it matches)
ref <- read_safetensors("~/chatterbox/outputs/llama_layers.safetensors", device = "cpu")
speaker_emb <- ref$speaker_emb$to(device = device)
cat(sprintf("Speaker embedding shape: %s\n", paste(speaker_emb$shape, collapse = "x")))

# Load conditioning debug output
cond_ref <- read_safetensors("~/chatterbox/outputs/cond_enc_debug.safetensors", device = "cpu")
cat("Loaded conditioning reference\n")

# Get text tokens from Python reference
text_tokens_py <- ref$text_tokens
cat(sprintf("Text tokens from Python: %s\n", paste(as.integer(text_tokens_py[1,]$cpu()), collapse = ", ")))

# Use Python text tokens directly (already has EOT and is doubled for CFG)
text_tokens_cfg <- text_tokens_py$to(device = device)
cat(sprintf("Text tokens shape: %s\n", paste(text_tokens_cfg$shape, collapse = "x")))

# BOS token
bos <- torch::torch_tensor(matrix(t3$config$start_speech_token, nrow = 1), device = device, dtype = torch::torch_long())
bos_cfg <- torch::torch_cat(list(bos, bos), dim = 1L)

# Create conditioning using Python's speech tokens
cond_prompt_speech_tokens <- cond_ref$cond_prompt_speech_tokens$to(device = device)
cat(sprintf("Speech tokens shape: %s\n", paste(cond_prompt_speech_tokens$shape, collapse = "x")))
cond <- t3_cond(speaker_emb, cond_prompt_speech_tokens = cond_prompt_speech_tokens, emotion_adv = 0.5)

# Prepare embeddings
cat("\n=== Preparing embeddings ===\n")
prep <- t3$prepare_input_embeds(cond, text_tokens_cfg, bos_cfg, cfg_weight = 0.5)
embeds <- prep$embeds
len_cond <- prep$len_cond

cat(sprintf("Input embeds shape: %s\n", paste(embeds$shape, collapse = "x")))
cat(sprintf("Conditioning length: %d\n", len_cond))
cat(sprintf("Embeds mean: %.6f, std: %.6f\n", embeds$mean()$item(), embeds$std()$item()))

# Run initial forward pass
cat("\n=== Running initial forward pass ===\n")
torch::with_no_grad({
  output <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)

  # Get logits from last position
  last_hidden <- output$last_hidden_state[, embeds$size(2), ]
  logits <- t3$speech_head$forward(last_hidden)

  cat(sprintf("Last hidden shape: %s\n", paste(last_hidden$shape, collapse = "x")))
  cat(sprintf("Logits shape: %s\n", paste(logits$shape, collapse = "x")))

  # CFG combination
  cond_logits <- logits[1, ]$unsqueeze(1)
  uncond_logits <- logits[2, ]$unsqueeze(1)
  cfg_weight <- 0.5
  combined <- cond_logits + cfg_weight * (cond_logits - uncond_logits)

  cat(sprintf("\nCond logits mean: %.6f, std: %.6f\n",
              cond_logits$mean()$item(), cond_logits$std()$item()))
  cat(sprintf("Uncond logits mean: %.6f, std: %.6f\n",
              uncond_logits$mean()$item(), uncond_logits$std()$item()))
  cat(sprintf("Combined (CFG) mean: %.6f, std: %.6f\n",
              combined$mean()$item(), combined$std()$item()))

  # Temperature
  temperature <- 0.8
  logits_temp <- combined / temperature

  # Convert to probs
  probs <- torch::nnf_softmax(logits_temp, dim = -1L)

  # Top tokens
  top_result <- torch::torch_topk(probs, k = 10L)
  top_probs <- top_result[[1]]
  top_indices <- top_result[[2]]

  cat(sprintf("\nTop 10 tokens (temp=%.1f):\n", temperature))
  for (i in 1:10) {
    token_id <- as.integer(top_indices[1, i]$item()) - 1L  # Convert back to 0-indexed
    prob <- top_probs[1, i]$item()
    token_name <- if (token_id == t3$config$stop_speech_token) "EOS" else as.character(token_id)
    cat(sprintf("  Token %s: %.6f\n", token_name, prob))
  }

  # Check EOS probability
  eos_prob <- probs[1, t3$config$stop_speech_token + 1L]$item()  # +1 for R indexing
  cat(sprintf("\nEOS token (%d) probability: %.6f\n", t3$config$stop_speech_token, eos_prob))
})

cat("\nDone.\n")
