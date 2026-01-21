#!/usr/bin/env r
# Compare R Llama layer outputs with Python reference

library(chatterbox)
library(torch)

# Source internal functions
source("~/chatterbox/R/t3.R")

# Load Python reference outputs
cat("Loading Python reference...\n")
ref_path <- "~/chatterbox/outputs/llama_layers.safetensors"
ref <- read_safetensors(ref_path, device = "cpu")

cat("Python outputs:\n")
for (name in names(ref)) {
  if (inherits(ref[[name]], "torch_tensor")) {
    shape <- paste(ref[[name]]$shape, collapse = "x")
    cat(sprintf("  %s: %s\n", name, shape))
  }
}

# Set up device
device <- if (cuda_is_available()) "cuda" else "cpu"
cat(sprintf("\nUsing device: %s\n", device))

# Load model
cat("Loading R model...\n")
model <- chatterbox(device)
model <- load_chatterbox(model)
t3 <- model$t3

# Get reference inputs
input_embeds_ref <- ref$input_embeds$to(device = device)
len_cond <- as.integer(ref$len_cond$item())
speaker_emb <- ref$speaker_emb$to(device = device)

cat(sprintf("\nInput embeds shape: %s\n", paste(input_embeds_ref$shape, collapse = "x")))
cat(sprintf("Conditioning length: %d\n", len_cond))

# Compare input embedding preparation
# First, let's compare the conditioning encoder
cat("\n=== Comparing conditioning ===\n")

# Load the detailed conditioning debug output
cond_ref_path <- "~/chatterbox/outputs/cond_enc_debug.safetensors"
if (file.exists(path.expand(cond_ref_path))) {
  cond_ref <- read_safetensors(cond_ref_path, device = "cpu")
  cond_prompt_speech_tokens <- cond_ref$cond_prompt_speech_tokens$to(device = device)
  cat(sprintf("Loaded cond_prompt_speech_tokens: %s\n", paste(cond_prompt_speech_tokens$shape, collapse = "x")))
} else {
  cond_prompt_speech_tokens <- NULL
  cat("WARNING: cond_enc_debug.safetensors not found, using NULL speech tokens\n")
}

# Create t3_cond from saved speaker embedding with speech tokens
cond <- t3_cond(speaker_emb, cond_prompt_speech_tokens = cond_prompt_speech_tokens, emotion_adv = 0.5)

# Process conditioning through the full prepare_conditioning method
# This will compute speech_emb from tokens and run perceiver
cond_emb_r <- t3$prepare_conditioning(cond)
cat(sprintf("R cond_enc output shape: %s\n", paste(cond_emb_r$shape, collapse = "x")))
cat(sprintf("R cond_enc output mean: %.6f, std: %.6f\n",
            cond_emb_r$mean()$item(), cond_emb_r$std()$item()))

# Extract conditioning part from Python embeds (first len_cond positions)
cond_emb_py <- input_embeds_ref[1, 1:len_cond, ]
cat(sprintf("Python cond embed mean: %.6f, std: %.6f\n",
            cond_emb_py$mean()$item(), cond_emb_py$std()$item()))

# Check if conditioning matches
cond_diff <- (cond_emb_r[1,,] - cond_emb_py)$abs()$max()$item()
cat(sprintf("Conditioning max diff: %.6f\n", cond_diff))

# First, let's compare R's own prepare_input_embeds against Python
cat("\n=== Comparing R prepare_input_embeds vs Python ===\n")

# Get text tokens from Python reference
text_tokens_ref <- ref$text_tokens$to(device = device)
cat(sprintf("Text tokens shape: %s\n", paste(text_tokens_ref$shape, collapse = "x")))

# BOS token
config <- t3$config
bos <- torch::torch_tensor(matrix(config$start_speech_token, nrow = 1), device = device, dtype = torch::torch_long())
bos_cfg <- torch::torch_cat(list(bos, bos), dim = 1L)

# Prepare R embeddings
prep_r <- t3$prepare_input_embeds(cond, text_tokens_ref, bos_cfg, cfg_weight = 0.5)
input_embeds_r <- prep_r$embeds
len_cond_r <- prep_r$len_cond

cat(sprintf("R input embeds shape: %s\n", paste(input_embeds_r$shape, collapse = "x")))
cat(sprintf("R conditioning length: %d\n", len_cond_r))
cat(sprintf("Python input embeds shape: %s\n", paste(input_embeds_ref$shape, collapse = "x")))
cat(sprintf("Python conditioning length: %d\n", len_cond))

# Compare
if (all(input_embeds_r$shape == input_embeds_ref$shape)) {
  diff_embeds <- (input_embeds_r - input_embeds_ref)$abs()
  cat(sprintf("Input embeds max diff: %.6f, mean diff: %.6f\n",
              diff_embeds$max()$item(), diff_embeds$mean()$item()))

  # Check by segment
  cat("\nPer-segment comparison:\n")

  # Conditioning
  cond_r <- input_embeds_r[1, 1:len_cond_r, ]
  cond_py <- input_embeds_ref[1, 1:len_cond, ]
  diff_cond <- (cond_r - cond_py)$abs()
  cat(sprintf("  Conditioning: max_diff=%.6f\n", diff_cond$max()$item()))

  # Text (after conditioning)
  text_len <- text_tokens_ref$size(2)
  text_start <- len_cond + 1
  text_end <- len_cond + text_len
  text_r <- input_embeds_r[1, text_start:text_end, ]
  text_py <- input_embeds_ref[1, text_start:text_end, ]
  diff_text <- (text_r - text_py)$abs()
  cat(sprintf("  Text embeddings: max_diff=%.6f\n", diff_text$max()$item()))

  # BOS (last position)
  bos_r <- input_embeds_r[1, input_embeds_r$size(2), ]
  bos_py <- input_embeds_ref[1, input_embeds_ref$size(2), ]
  diff_bos <- (bos_r - bos_py)$abs()
  cat(sprintf("  BOS embedding: max_diff=%.6f\n", diff_bos$max()$item()))

  # Check CFG path (uncond - second batch element)
  cat("\nCFG uncond path comparison:\n")
  text_r_uncond <- input_embeds_r[2, text_start:text_end, ]
  text_py_uncond <- input_embeds_ref[2, text_start:text_end, ]
  diff_text_uncond <- (text_r_uncond - text_py_uncond)$abs()
  cat(sprintf("  Text embeddings (uncond): max_diff=%.6f\n", diff_text_uncond$max()$item()))

  # Check if uncond text is zeroed
  cat(sprintf("  R uncond text mean: %.6f (should be ~0)\n", text_r_uncond$mean()$item()))
  cat(sprintf("  Python uncond text mean: %.6f\n", text_py_uncond$mean()$item()))
} else {
  cat("Shape mismatch - cannot compare directly\n")
}

# Now let's run through Llama with the EXACT Python input embeddings
cat("\n=== Running Llama with Python input embeds ===\n")

torch::with_no_grad({
  # Forward through Llama with output_hidden_states=TRUE
  output <- t3$tfmr$forward(
    inputs_embeds = input_embeds_ref,
    use_cache = FALSE,
    output_hidden_states = TRUE
  )

  # Compare layer outputs
  cat("\nLayer-by-layer comparison (cond path, batch 1):\n")

  # output$hidden_states contains all layer outputs
  if (!is.null(output$hidden_states) && length(output$hidden_states) > 0) {
    for (i in seq_along(output$hidden_states)) {
      layer_name <- sprintf("layer_%d", i - 1)  # 0-indexed layer names
      h <- output$hidden_states[[i]]
      h_cond <- h[1,,]  # First batch element

      cat(sprintf("R %s: mean=%.6f, std=%.6f\n",
                  layer_name, h_cond$mean()$item(), h_cond$std()$item()))

      # Compare with Python reference
      if (layer_name %in% names(ref)) {
        py_layer <- ref[[layer_name]]$to(device = device)
        py_cond <- py_layer[1,,]

        diff <- (h_cond - py_cond)$abs()
        cat(sprintf("   Python: mean=%.6f, std=%.6f, max_diff=%.6f\n",
                    py_cond$mean()$item(), py_cond$std()$item(), diff$max()$item()))
      }
    }
  } else {
    cat("WARNING: hidden_states is NULL or empty\n")
  }

  # Final hidden state
  cat("\n=== Final output ===\n")
  last_hidden_r <- output$last_hidden_state
  last_hidden_r_cond <- last_hidden_r[1,,]
  cat(sprintf("R last_hidden_state (cond): mean=%.6f, std=%.6f\n",
              last_hidden_r_cond$mean()$item(), last_hidden_r_cond$std()$item()))

  last_hidden_py <- ref$last_hidden_state$to(device = device)
  last_hidden_py_cond <- last_hidden_py[1,,]
  cat(sprintf("Python last_hidden_state (cond): mean=%.6f, std=%.6f\n",
              last_hidden_py_cond$mean()$item(), last_hidden_py_cond$std()$item()))

  diff <- (last_hidden_r_cond - last_hidden_py_cond)$abs()
  cat(sprintf("Last hidden max diff: %.6f, mean diff: %.6f\n",
              diff$max()$item(), diff$mean()$item()))

  # Logits
  last_pos_r <- last_hidden_r[, last_hidden_r$size(2), ]
  logits_r <- t3$speech_head$forward(last_pos_r)
  logits_r_cond <- logits_r[1, ]
  cat(sprintf("\nR logits (cond): mean=%.6f, std=%.6f\n",
              logits_r_cond$mean()$item(), logits_r_cond$std()$item()))

  logits_py <- ref$logits$to(device = device)
  logits_py_cond <- logits_py[1, ]
  cat(sprintf("Python logits (cond): mean=%.6f, std=%.6f\n",
              logits_py_cond$mean()$item(), logits_py_cond$std()$item()))

  diff <- (logits_r_cond - logits_py_cond)$abs()
  cat(sprintf("Logits max diff: %.6f, mean diff: %.6f\n",
              diff$max()$item(), diff$mean()$item()))
})

cat("\nDone.\n")
