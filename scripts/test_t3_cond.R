#!/usr/bin/env r
# Test R T3 conditioning against Python reference

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/llama.R")
source("~/chatterbox/R/t3.R")

cat("Loading Python reference...\n")
ref <- read_safetensors("~/chatterbox/outputs/t3_steps.safetensors")
cat("Reference keys:", paste(names(ref), collapse = ", "), "\n\n")

# Load T3 weights from HuggingFace cache
cat("Loading T3 weights...\n")
weights <- read_safetensors("~/.cache/chatterbox/ResembleAI--chatterbox/t3_cfg.safetensors")
cat(sprintf("Loaded %d weight tensors\n\n", length(weights)))

# Reference values
py_speaker_emb <- ref$speaker_emb
py_prompt_tokens <- ref$prompt_tokens
py_text_tokens <- ref$text_tokens
py_cond <- ref$cond
py_text_emb_tok_only <- ref$text_emb_tok_only
py_text_pos_emb <- ref$text_pos_emb
py_text_emb <- ref$text_emb  # token + position
py_speech_start_emb <- ref$speech_start_emb
py_input_embeds <- ref$input_embeds
start_speech_token <- as.integer(ref$start_speech_token$item())
stop_speech_token <- as.integer(ref$stop_speech_token$item())

cat("=== Python Reference ===\n")
cat(sprintf("Speaker embedding: %s\n", paste(dim(py_speaker_emb), collapse="x")))
cat(sprintf("Prompt tokens: %s\n", paste(dim(py_prompt_tokens), collapse="x")))
cat(sprintf("Text tokens: %s, values: %s\n",
            paste(dim(py_text_tokens), collapse="x"),
            paste(as.integer(py_text_tokens), collapse=", ")))
cat(sprintf("Conditioning: %s\n", paste(dim(py_cond), collapse="x")))
cat(sprintf("Text token emb: %s\n", paste(dim(py_text_emb_tok_only), collapse="x")))
cat(sprintf("Text pos emb: %s\n", paste(dim(py_text_pos_emb), collapse="x")))
cat(sprintf("Text embedding (tok+pos): %s\n", paste(dim(py_text_emb), collapse="x")))
cat(sprintf("Speech start embedding: %s\n", paste(dim(py_speech_start_emb), collapse="x")))
cat(sprintf("Input embeds: %s\n", paste(dim(py_input_embeds), collapse="x")))
cat(sprintf("start_speech_token: %d\n", start_speech_token))
cat(sprintf("stop_speech_token: %d\n", stop_speech_token))

# ============================================================================
# Create and load T3 model
# ============================================================================
cat("\n=== Creating R T3 Model ===\n")
config <- t3_config_english()
cat("Config created\n")

model <- t3_model(config)
cat("Model created\n")

# Load weights
model <- load_t3_weights(model, weights)
model$eval()
cat("Weights loaded\n")

# ============================================================================
# Step 1: Text embeddings
# ============================================================================
cat("\n=== Step 1: Text Embeddings ===\n")

torch::with_no_grad({
  # Python uses 0-indexed tokens, R nn_embedding expects 1-indexed
  r_text_emb_tok_only <- model$text_emb$forward(py_text_tokens$add(1L))
})
cat(sprintf("R text token emb shape: %s\n", paste(dim(r_text_emb_tok_only), collapse="x")))

# Compare token-only embeddings
diff_text_tok <- (r_text_emb_tok_only - py_text_emb_tok_only)$abs()$max()$item()
cat(sprintf("Text token embedding diff: %.6f\n", diff_text_tok))

# Get position embeddings
torch::with_no_grad({
  r_text_pos_emb <- model$text_pos_emb$forward(py_text_tokens)
})
cat(sprintf("R text pos emb shape: %s\n", paste(dim(r_text_pos_emb), collapse="x")))

# Compare position embeddings
diff_text_pos <- (r_text_pos_emb - py_text_pos_emb)$abs()$max()$item()
cat(sprintf("Text position embedding diff: %.6f\n", diff_text_pos))

# Combined text embedding
r_text_emb_with_pos <- r_text_emb_tok_only + r_text_pos_emb
diff_text_combined <- (r_text_emb_with_pos - py_text_emb)$abs()$max()$item()
cat(sprintf("Text combined (tok+pos) diff: %.6f\n", diff_text_combined))

# ============================================================================
# Step 2: Speech start token embedding
# ============================================================================
cat("\n=== Step 2: Speech Start Token Embedding ===\n")

torch::with_no_grad({
  start_token <- torch::torch_tensor(matrix(start_speech_token, nrow = 1),
    dtype = torch::torch_long())
  r_speech_emb <- model$speech_emb$forward(start_token$add(1L))
  r_speech_emb_with_pos <- r_speech_emb + model$speech_pos_emb$get_fixed_embedding(0L)
})
cat(sprintf("R speech start embedding shape: %s\n", paste(dim(r_speech_emb_with_pos), collapse="x")))

diff_speech_emb <- (r_speech_emb_with_pos - py_speech_start_emb)$abs()$max()$item()
cat(sprintf("Speech start embedding diff: %.6f\n", diff_speech_emb))

# ============================================================================
# Step 3: T3 Conditioning
# ============================================================================
cat("\n=== Step 3: T3 Conditioning ===\n")

# Build T3 conditioning
cond <- t3_cond(
  speaker_emb = py_speaker_emb,
  cond_prompt_speech_tokens = py_prompt_tokens,
  emotion_adv = 0.5
)

torch::with_no_grad({
  r_cond <- model$prepare_conditioning(cond)
})
cat(sprintf("R conditioning shape: %s\n", paste(dim(r_cond), collapse="x")))
cat(sprintf("R conditioning mean: %.6f, std: %.6f\n",
            r_cond$mean()$item(), r_cond$std()$item()))
cat(sprintf("Py conditioning mean: %.6f, std: %.6f\n",
            py_cond$mean()$item(), py_cond$std()$item()))

diff_cond <- (r_cond - py_cond)$abs()$max()$item()
mean_diff_cond <- (r_cond - py_cond)$abs()$mean()$item()
cat(sprintf("Conditioning max diff: %.6f\n", diff_cond))
cat(sprintf("Conditioning mean diff: %.6f\n", mean_diff_cond))

# Check individual components
cat("\n--- Conditioning Components ---\n")
# cond is: [speaker (1), clap (0), perceiver (32), emotion (1)] = 34 total

# Speaker component (position 1)
r_spkr <- r_cond[, 1, ]
py_spkr <- py_cond[, 1, ]
diff_spkr <- (r_spkr - py_spkr)$abs()$max()$item()
cat(sprintf("Speaker projection diff: %.6f\n", diff_spkr))

# Perceiver output (positions 2-33)
r_perceiver <- r_cond[, 2:33, ]
py_perceiver <- py_cond[, 2:33, ]
diff_perceiver <- (r_perceiver - py_perceiver)$abs()$max()$item()
mean_perceiver <- (r_perceiver - py_perceiver)$abs()$mean()$item()
cat(sprintf("Perceiver output max diff: %.6f\n", diff_perceiver))
cat(sprintf("Perceiver output mean diff: %.6f\n", mean_perceiver))
cat(sprintf("Perceiver R stats: mean=%.4f, std=%.4f\n",
            r_perceiver$mean()$item(), r_perceiver$std()$item()))
cat(sprintf("Perceiver Py stats: mean=%.4f, std=%.4f\n",
            py_perceiver$mean()$item(), py_perceiver$std()$item()))

# Emotion component (position 34)
r_emotion <- r_cond[, 34, ]
py_emotion <- py_cond[, 34, ]
diff_emotion <- (r_emotion - py_emotion)$abs()$max()$item()
cat(sprintf("Emotion projection diff: %.6f\n", diff_emotion))

# ============================================================================
# Step 4: Full input embeddings
# ============================================================================
cat("\n=== Step 4: Full Input Embeddings ===\n")

torch::with_no_grad({
  r_input_embeds <- torch::torch_cat(list(r_cond, r_text_emb_with_pos, r_speech_emb_with_pos), dim = 2)
})
cat(sprintf("R input embeds shape: %s\n", paste(dim(r_input_embeds), collapse="x")))

diff_input <- (r_input_embeds - py_input_embeds)$abs()$max()$item()
mean_diff_input <- (r_input_embeds - py_input_embeds)$abs()$mean()$item()
cat(sprintf("Input embeds max diff: %.6f\n", diff_input))
cat(sprintf("Input embeds mean diff: %.6f\n", mean_diff_input))

# ============================================================================
# Summary
# ============================================================================
cat("\n=== Summary ===\n")
cat(sprintf("Text token embedding:      %.6f\n", diff_text_tok))
cat(sprintf("Text position embedding:   %.6f\n", diff_text_pos))
cat(sprintf("Text combined (tok+pos):   %.6f\n", diff_text_combined))
cat(sprintf("Speech start embedding:    %.6f\n", diff_speech_emb))
cat(sprintf("Speaker projection:        %.6f\n", diff_spkr))
cat(sprintf("Perceiver output:          %.6f\n", diff_perceiver))
cat(sprintf("Emotion projection:        %.6f\n", diff_emotion))
cat(sprintf("Full conditioning:         %.6f\n", diff_cond))
cat(sprintf("Full input embeddings:     %.6f\n", diff_input))

# Check if all pass threshold
threshold <- 1e-4
all_pass <- all(c(
  diff_text_tok < threshold,
  diff_text_pos < threshold,
  diff_text_combined < threshold,
  diff_speech_emb < threshold,
  diff_spkr < threshold,
  diff_emotion < threshold,
  diff_cond < threshold
))

if (all_pass) {
  cat("\nAll components match within threshold\n")
} else {
  cat(sprintf("\nSome components exceed threshold %.6f\n", threshold))
  if (diff_text_pos > threshold) {
    cat("  - Text position embedding needs investigation\n")
  }
  if (diff_perceiver > threshold) {
    cat("  - Perceiver needs investigation\n")
  }
}

cat("\nDone.\n")
