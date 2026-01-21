#!/usr/bin/env Rscript
# Debug T3 conditioning encoder

rhydrogen::load_all()

# Load model
device <- "cuda"
model <- chatterbox(device)
model <- load_chatterbox(model)

# Reference audio
ref_audio <- "/home/troy/cornball_media/LilCasey/Casey_voice_samples/ShortCasey.wav"

# Create voice embedding
voice <- create_voice_embedding(model, ref_audio)

cat("=== Voice Embedding ===\n")
cat("Speaker embedding shape:", paste(dim(voice$ve_embedding), collapse = "x"), "\n")
cat("Speaker embedding mean:", voice$ve_embedding$mean()$item(), "\n")

# Create T3 conditioning - NOW with cond_prompt_speech_tokens
cat("\n=== T3 Conditioning ===\n")
cond <- t3_cond(
  speaker_emb = voice$ve_embedding,
  cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
  emotion_adv = 0.5
)

cat("Speaker emb in cond:", paste(dim(cond$speaker_emb), collapse = "x"), "\n")
cat("Emotion adv:", cond$emotion_adv, "\n")
if (is.null(cond$cond_prompt_speech_tokens)) {
  cat("cond_prompt_speech_tokens: NULL\n")
} else {
  cat("cond_prompt_speech_tokens shape:", paste(dim(cond$cond_prompt_speech_tokens), collapse = "x"), "\n")
  cat("cond_prompt_speech_tokens first 10:", paste(as.integer(cond$cond_prompt_speech_tokens[1, 1:10]$cpu()), collapse = ", "), "\n")
}
cat("cond_prompt_speech_emb is NULL:", is.null(cond$cond_prompt_speech_emb), "\n")

# Move to device
cond <- t3_cond_to_device(cond, device)

# Now run the conditioning encoder directly
cat("\n=== T3 Conditioning Encoder ===\n")

torch::with_no_grad({
  # Get the config
  config <- model$t3$config
  cat("Config n_channels:", config$n_channels, "\n")
  cat("Config speaker_embed_size:", config$speaker_embed_size, "\n")
  cat("Config use_perceiver_resampler:", config$use_perceiver_resampler, "\n")

  # Run the conditioning encoder
  cond_enc <- model$t3$cond_enc

  # Get speaker embedding shape
  batch_size <- cond$speaker_emb$size(1)
  cat("Batch size:", batch_size, "\n")

  # Speaker embedding projection
  spkr_flat <- cond$speaker_emb$view(c(-1, config$speaker_embed_size))
  cat("Speaker emb flattened shape:", paste(dim(spkr_flat), collapse = "x"), "\n")

  cond_spkr <- cond_enc$spkr_enc$forward(spkr_flat)
  cat("After spkr_enc shape:", paste(dim(cond_spkr), collapse = "x"), "\n")

  cond_spkr <- cond_spkr$unsqueeze(2)
  cat("After unsqueeze shape:", paste(dim(cond_spkr), collapse = "x"), "\n")

  # Empty tensor for unused conditioning
  empty <- torch::torch_zeros(c(batch_size, 0L, config$n_channels), device = device)
  cat("Empty tensor shape:", paste(dim(empty), collapse = "x"), "\n")

  # CLAP not implemented
  cond_clap <- empty

  # Conditional prompt speech embeddings
  cond_prompt_speech_emb <- cond$cond_prompt_speech_emb
  if (is.null(cond_prompt_speech_emb)) {
    cond_prompt_speech_emb <- empty
    cat("No cond_prompt_speech_emb, using empty\n")
  } else {
    cat("cond_prompt_speech_emb shape:", paste(dim(cond_prompt_speech_emb), collapse = "x"), "\n")
    if (!is.null(cond_enc$perceiver)) {
      cat("Applying perceiver resampler...\n")
      cond_prompt_speech_emb <- cond_enc$perceiver$forward(cond_prompt_speech_emb)
      cat("After perceiver shape:", paste(dim(cond_prompt_speech_emb), collapse = "x"), "\n")
    }
  }

  # Emotion control
  cond_emotion_adv <- empty
  if (!is.null(cond_enc$emotion_adv_fc) && !is.null(cond$emotion_adv)) {
    cat("Applying emotion embedding...\n")
    emotion_val <- cond$emotion_adv
    if (!inherits(emotion_val, "torch_tensor")) {
      emotion_val <- torch::torch_tensor(emotion_val, device = device)
    }
    emotion_val <- emotion_val$view(c(-1, 1L, 1L))
    cat("Emotion val shape:", paste(dim(emotion_val), collapse = "x"), "\n")
    cond_emotion_adv <- cond_enc$emotion_adv_fc$forward(emotion_val)
    cat("After emotion_adv_fc shape:", paste(dim(cond_emotion_adv), collapse = "x"), "\n")
  }

  # Concatenate all conditioning
  cat("\n=== Concatenation ===\n")
  cat("cond_spkr shape:", paste(dim(cond_spkr), collapse = "x"), "\n")
  cat("cond_clap shape:", paste(dim(cond_clap), collapse = "x"), "\n")
  cat("cond_prompt_speech_emb shape:", paste(dim(cond_prompt_speech_emb), collapse = "x"), "\n")
  cat("cond_emotion_adv shape:", paste(dim(cond_emotion_adv), collapse = "x"), "\n")

  final_cond <- torch::torch_cat(list(cond_spkr, cond_clap, cond_prompt_speech_emb, cond_emotion_adv), dim = 2)
  cat("Final conditioning shape:", paste(dim(final_cond), collapse = "x"), "\n")
})

# Compare with Python
cat("\n=== Compare with Python ===\n")
# Load Python debug
debug_path <- "/home/troy/chatterbox/outputs/t3_debug.safetensors"
py_debug <- read_safetensors(debug_path, device = "cpu")
cat("Python initial_embeds shape:", paste(dim(py_debug$initial_embeds), collapse = "x"), "\n")
cat("Python speaker_emb shape:", paste(dim(py_debug$speaker_emb), collapse = "x"), "\n")
