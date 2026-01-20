#!/usr/bin/env Rscript
# Compare R T3 output against Python reference

rhydrogen::load_all()

# Load Python reference
ref_path <- "/home/troy/chatteRbox/outputs/t3_reference.safetensors"
message("Loading Python reference from: ", ref_path)
ref <- read_safetensors(ref_path, device = "cpu")

cat("\n=== Python Reference ===\n")
cat("Text tokens:", paste(as.integer(ref$text_tokens), collapse = ", "), "\n")
cat("Text tokens with SOT/EOT:", paste(as.integer(ref$text_tokens_with_sot_eot), collapse = ", "), "\n")
cat("Speaker embedding shape:", paste(dim(ref$speaker_emb), collapse = "x"), "\n")
cat("Speaker embedding mean:", mean(as.numeric(ref$speaker_emb)), "\n")
cat("Speaker embedding std:", sd(as.numeric(ref$speaker_emb)), "\n")

if ("cond_prompt_speech_tokens" %in% names(ref)) {
  cond_tokens <- ref$cond_prompt_speech_tokens
  cat("Cond prompt tokens shape:", paste(dim(cond_tokens), collapse = "x"), "\n")
  cat("Cond prompt tokens (first 20):",
      paste(as.integer(cond_tokens[1, 1:20]$cpu()), collapse = ", "), "\n")
}

py_speech <- as.integer(ref$speech_tokens$cpu())
cat("\nPython speech tokens (first 50):",
    paste(py_speech[1:min(50, length(py_speech))], collapse = ", "), "\n")
cat("Python speech tokens count:", length(py_speech), "\n")
cat("Python speech tokens min:", min(py_speech), "\n")
cat("Python speech tokens max:", max(py_speech), "\n")

# Now generate with R
cat("\n=== Loading R Model ===\n")
device <- "cuda"
model <- chatterbox(device)
model <- load_chatterbox(model)

# Use same reference audio
ref_audio <- "/home/troy/cornball_media/LilCasey/Casey_voice_samples/ShortCasey.wav"
text <- "Hello, this is a test of the text to speech system."

cat("\n=== R Text Tokenization ===\n")
# Tokenize text
text_tokens <- tokenize_text(model$tokenizer, text)
cat("R text tokens:", paste(text_tokens, collapse = ", "), "\n")

# Compare with Python
py_text_tokens <- as.integer(ref$text_tokens)
if (identical(text_tokens, py_text_tokens)) {
  cat("TEXT TOKENS MATCH\n")
} else {
  cat("TEXT TOKENS DIFFER\n")
  cat("  Python length:", length(py_text_tokens), "\n")
  cat("  R length:", length(text_tokens), "\n")
  # Find differences
  min_len <- min(length(text_tokens), length(py_text_tokens))
  diffs <- which(text_tokens[1:min_len] != py_text_tokens[1:min_len])
  if (length(diffs) > 0) {
    cat("  First difference at position:", diffs[1], "\n")
    cat("    Python:", py_text_tokens[diffs[1]], "\n")
    cat("    R:", text_tokens[diffs[1]], "\n")
  }
}

cat("\n=== R Voice Embedding ===\n")
voice <- create_voice_embedding(model, ref_audio)
r_speaker_emb <- voice$ve_embedding

cat("R speaker embedding shape:", paste(dim(r_speaker_emb), collapse = "x"), "\n")
cat("R speaker embedding mean:", mean(as.numeric(r_speaker_emb$cpu())), "\n")
cat("R speaker embedding std:", sd(as.numeric(r_speaker_emb$cpu())), "\n")

# Compare speaker embeddings
py_speaker_emb <- ref$speaker_emb$to(device = device)
emb_diff <- (r_speaker_emb - py_speaker_emb)$abs()
cat("Speaker embedding max diff:", emb_diff$max()$item(), "\n")
cat("Speaker embedding mean diff:", emb_diff$mean()$item(), "\n")

if (emb_diff$max()$item() < 0.01) {
  cat("SPEAKER EMBEDDINGS MATCH (within tolerance)\n")
} else {
  cat("SPEAKER EMBEDDINGS DIFFER SIGNIFICANTLY\n")
}

cat("\n=== R T3 Inference ===\n")
# Set seed for reproducibility
set.seed(42)
torch::torch_manual_seed(42L)
if (torch::cuda_is_available()) {
  # Note: torch_cuda_manual_seed doesn't exist in R torch, using global seed
}

text_tokens_tensor <- torch::torch_tensor(text_tokens, dtype = torch::torch_long())$unsqueeze(1)$to(device = device)

cond <- t3_cond(
  speaker_emb = r_speaker_emb,
  emotion_adv = 0.5
)

torch::with_no_grad({
  speech_tokens <- t3_inference(
    model = model$t3,
    cond = cond,
    text_tokens = text_tokens_tensor,
    cfg_weight = 0.5,
    temperature = 0.8,
    top_p = 0.9,
    min_p = 0.05,
    repetition_penalty = 1.2
  )
})

r_speech_tokens <- as.integer(speech_tokens$cpu())
cat("R speech tokens count:", length(r_speech_tokens), "\n")
cat("R speech tokens (first 50):",
    paste(r_speech_tokens[1:min(50, length(r_speech_tokens))], collapse = ", "), "\n")
cat("R speech tokens min:", min(r_speech_tokens), "\n")
cat("R speech tokens max:", max(r_speech_tokens), "\n")

# Compare with Python
py_speech_tokens <- as.integer(ref$speech_tokens)
cat("\n=== Speech Token Comparison ===\n")
cat("Python count:", length(py_speech_tokens), "\n")
cat("R count:", length(r_speech_tokens), "\n")

# Token range comparison
cat("Python range: [", min(py_speech_tokens), ", ", max(py_speech_tokens), "]\n", sep = "")
cat("R range: [", min(r_speech_tokens), ", ", max(r_speech_tokens), "]\n", sep = "")

# Distribution comparison
py_unique <- length(unique(py_speech_tokens))
r_unique <- length(unique(r_speech_tokens))
cat("Python unique tokens:", py_unique, "\n")
cat("R unique tokens:", r_unique, "\n")

# Check if tokens are in valid range (should be < 6561 for speech)
invalid_r <- sum(r_speech_tokens >= 6561)
invalid_py <- sum(py_speech_tokens >= 6561)
cat("Python invalid tokens (>= 6561):", invalid_py, "\n")
cat("R invalid tokens (>= 6561):", invalid_r, "\n")

# Due to sampling, exact match is not expected, but check overlap
overlap <- length(intersect(py_speech_tokens, r_speech_tokens))
cat("Token overlap:", overlap, "tokens appear in both\n")

cat("\n=== Summary ===\n")
if (identical(text_tokens, py_text_tokens)) {
  cat("Text tokenization: PASS\n")
} else {
  cat("Text tokenization: FAIL\n")
}

if (emb_diff$max()$item() < 0.01) {
  cat("Voice embedding: PASS\n")
} else {
  cat("Voice embedding: FAIL\n")
}

# For speech tokens, check if distribution looks reasonable
if (min(r_speech_tokens) >= 0 && max(r_speech_tokens) < 6561 && length(r_speech_tokens) > 10) {
  cat("Speech token generation: PLAUSIBLE (different due to sampling)\n")
} else {
  cat("Speech token generation: SUSPICIOUS - check range and count\n")
}
