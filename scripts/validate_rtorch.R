#!/usr/bin/env r
# Quick component validation for Rtorch backend
# Compares intermediate outputs against Python reference safetensors

library(chatterbox)

cat("=== Rtorch Component Validation ===\n\n")

# Helper
compare <- function(label, r_tensor, ref_tensor) {
  diff <- (r_tensor - ref_tensor)$abs()$max()$item()
  r_mean <- r_tensor$mean()$item()
  ref_mean <- ref_tensor$mean()$item()
  r_std <- r_tensor$std()$item()
  ref_std <- ref_tensor$std()$item()
  status <- if (diff < 0.001) "OK" else if (diff < 0.1) "WARN" else "FAIL"
  cat(sprintf("  [%s] %s: max_diff=%.6f  R_mean=%.6f  Ref_mean=%.6f  R_std=%.4f  Ref_std=%.4f\n",
              status, label, diff, r_mean, ref_mean, r_std, ref_std))
  diff
}

# ============================================================================
# 1. Voice Encoder (mel + LSTM)
# ============================================================================
cat("--- Voice Encoder ---\n")
ve_ref <- chatterbox:::read_safetensors("outputs/voice_encoder_reference.safetensors")
cat(sprintf("  Ref keys: %s\n", paste(names(ve_ref), collapse = ", ")))

# Load model (just voice encoder)
cat("  Loading model...\n")
model <- chatterbox("cpu")
model <- load_chatterbox(model)
cat("  Model loaded.\n")

# Create voice embedding and check
voice <- create_voice_embedding(model, "inst/audio/jfk.wav")
cat(sprintf("  Voice embedding shape: %s\n", paste(dim(voice$embedding), collapse = "x")))

if ("official_embedding" %in% names(ve_ref)) {
  compare("speaker_embedding", voice$embedding, ve_ref$official_embedding)
}

# ============================================================================
# 2. S3 Tokenizer
# ============================================================================
cat("\n--- S3 Tokenizer ---\n")
s3_ref <- chatterbox:::read_safetensors("outputs/s3tokenizer_steps.safetensors")
cat(sprintf("  Ref keys: %s\n", paste(names(s3_ref), collapse = ", ")))

if ("prompt_speech_tokens" %in% names(s3_ref)) {
  ref_tokens <- s3_ref$prompt_speech_tokens
  cat(sprintf("  Ref tokens shape: %s, first 10: %s\n",
              paste(dim(ref_tokens), collapse = "x"),
              paste(as.integer(ref_tokens[1, 1:min(10, dim(ref_tokens)[2])]), collapse = ", ")))

  # Check voice embedding prompt tokens
  if (!is.null(voice$ref_dict$prompt_token)) {
    r_tokens <- voice$ref_dict$prompt_token
    cat(sprintf("  R tokens shape: %s, first 10: %s\n",
                paste(dim(r_tokens), collapse = "x"),
                paste(as.integer(r_tokens[1, 1:min(10, dim(r_tokens)[2])]), collapse = ", ")))
  }
}

# ============================================================================
# 3. T3 Token Generation (quick check)
# ============================================================================
cat("\n--- T3 Token Generation ---\n")
text <- "Hello."
cat(sprintf("  Text: \"%s\"\n", text))

Rtorch::with_no_grad({
  result <- generate(model, text, voice)
  audio <- result$audio
  sr <- result$sample_rate
})

cat(sprintf("  Generated: %.2f seconds audio\n", length(audio) / sr))
cat(sprintf("  Audio stats: mean=%.6f, std=%.6f, min=%.4f, max=%.4f\n",
            mean(audio), sd(audio), min(audio), max(audio)))

# ============================================================================
# 4. HiFiGAN Reference Check
# ============================================================================
cat("\n--- HiFiGAN ---\n")
hifi_ref <- chatterbox:::read_safetensors("outputs/hifigan_reference.safetensors")
cat(sprintf("  Ref keys: %s\n", paste(names(hifi_ref), collapse = ", ")))

if ("input_mel" %in% names(hifi_ref) && "output_audio" %in% names(hifi_ref)) {
  cat(sprintf("  Input mel: %s\n", paste(dim(hifi_ref$input_mel), collapse = "x")))
  cat(sprintf("  Output audio: %s\n", paste(dim(hifi_ref$output_audio), collapse = "x")))

  # Run vocoder on reference mel
  cat("  Running HiFiGAN on reference mel...\n")
  Rtorch::with_no_grad({
    r_audio <- model$s3gen$mel2wav$forward(hifi_ref$input_mel)
  })
  cat(sprintf("  R audio: %s\n", paste(dim(r_audio), collapse = "x")))
  if (prod(dim(r_audio)) > 0 && prod(dim(hifi_ref$output_audio)) > 0) {
    min_len <- min(dim(r_audio)[length(dim(r_audio))], dim(hifi_ref$output_audio)[length(dim(hifi_ref$output_audio))])
    if (r_audio$dim() == 2) {
      compare("hifigan_audio", r_audio[, 1:min_len], hifi_ref$output_audio[, 1:min_len])
    } else {
      compare("hifigan_audio", r_audio[1, 1, 1:min_len], hifi_ref$output_audio[1, 1, 1:min_len])
    }
  }
}

# ============================================================================
# 5. CFM Decoder Reference Check
# ============================================================================
cat("\n--- CFM Decoder ---\n")
cfm_ref <- chatterbox:::read_safetensors("outputs/cfm_steps.safetensors")
cat(sprintf("  Ref keys: %s\n", paste(names(cfm_ref), collapse = ", ")))

cat("\nDone.\n")
