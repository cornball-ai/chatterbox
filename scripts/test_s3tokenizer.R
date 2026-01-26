#!/usr/bin/env r
# Test R S3 Tokenizer against Python reference

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")
source("~/chatterbox/R/s3tokenizer.R")

cat("Loading Python reference...\n")
s3_ref <- read_safetensors("~/chatterbox/outputs/s3tokenizer_steps.safetensors")
cat("Reference keys:", paste(names(s3_ref), collapse = ", "), "\n\n")

# Load weights
cat("Loading S3 tokenizer weights...\n")
weights <- read_safetensors("~/chatterbox/outputs/s3tokenizer_weights.safetensors")
cat(sprintf("Loaded %d weight tensors\n\n", length(weights)))

# Get audio
wav_16k <- s3_ref$wav_16k
cat(sprintf("Audio: %d samples at 16kHz (%.2fs)\n", length(wav_16k), length(wav_16k) / 16000))

# Expected tokens from Python
py_tokens <- s3_ref$prompt_speech_tokens
cat(sprintf("Python tokens: shape=%s\n", paste(dim(py_tokens), collapse = "x")))
cat(sprintf("  First 10: %s\n", paste(as.integer(py_tokens[1, 1:10]), collapse = ", ")))

# ============================================================================
# Create R tokenizer and load weights
# ============================================================================
cat("\n=== Creating R S3 Tokenizer ===\n")
config <- s3_tokenizer_config()
tokenizer <- s3_tokenizer(config)
cat("Tokenizer created\n")

# Load weights
tokenizer <- load_s3tokenizer_weights(tokenizer, weights)
tokenizer$eval()
cat("Weights loaded\n")

# ============================================================================
# Step 1: Mel spectrogram
# ============================================================================
cat("\n=== Step 1: Mel Spectrogram ===\n")

# Convert audio to tensor
audio_t <- torch::torch_tensor(as.numeric(wav_16k))$unsqueeze(1)
cat(sprintf("Audio tensor shape: %s\n", paste(dim(audio_t), collapse = "x")))

# Compute mel
torch::with_no_grad({
        mel <- tokenizer$log_mel_spectrogram(audio_t)
    })
cat(sprintf("Mel shape: %s\n", paste(dim(mel), collapse = "x")))
cat(sprintf("Mel stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
        mel$mean()$item(), mel$std()$item(), mel$min()$item(), mel$max()$item()))

# ============================================================================
# Step 2: Full tokenization
# ============================================================================
cat("\n=== Step 2: Full Tokenization ===\n")

max_len <- 150L# Same as Python
torch::with_no_grad({
        # Truncate mel for max_len
        mel_trunc <- mel[,, 1:min(mel$size(3), max_len * 4L)]
        cat(sprintf("Truncated mel shape: %s\n", paste(dim(mel_trunc), collapse = "x")))

        mel_lens <- torch::torch_tensor(mel_trunc$size(3))$unsqueeze(1)

        # Quantize
        result <- tokenizer$quantize(mel_trunc, mel_lens)
        r_tokens <- result$tokens
        r_lens <- result$lens
    })

cat(sprintf("R tokens shape: %s\n", paste(dim(r_tokens), collapse = "x")))
cat(sprintf("R token lens: %s\n", as.integer(r_lens)))
cat(sprintf("R first 10: %s\n", paste(as.integer(r_tokens[1, 1:10]), collapse = ", ")))
cat(sprintf("R last 10: %s\n", paste(as.integer(r_tokens[1, (dim(r_tokens)[2] - 9) :dim(r_tokens)[2]]), collapse = ", ")))

# Compare with Python
py_first10 <- as.integer(py_tokens[1, 1:10])
r_first10 <- as.integer(r_tokens[1, 1:10])
cat(sprintf("\nPy first 10: %s\n", paste(py_first10, collapse = ", ")))
cat(sprintf("R  first 10: %s\n", paste(r_first10, collapse = ", ")))

# Check if they match
matches <- sum(as.integer(r_tokens) == as.integer(py_tokens))
total <- prod(dim(py_tokens))
cat(sprintf("\nToken match: %d / %d (%.1f%%)\n", matches, total, 100 * matches / total))

# Token statistics
cat(sprintf("\nPython token stats: min=%d, max=%d, unique=%d\n",
        min(as.integer(py_tokens)), max(as.integer(py_tokens)),
        length(unique(as.integer(py_tokens)))))
cat(sprintf("R token stats: min=%d, max=%d, unique=%d\n",
        min(as.integer(r_tokens)), max(as.integer(r_tokens)),
        length(unique(as.integer(r_tokens)))))

cat("\nDone.\n")

