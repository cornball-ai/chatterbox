#!/usr/bin/env r
# Compare R mel spectrogram against Python reference

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")
source("~/chatterbox/R/voice_encoder.R")

cat("=== Loading Python reference ===\n")
ref <- read_safetensors("~/chatterbox/outputs/mel_reference.safetensors")

cat("Python reference contents:\n")
for (k in names(ref)) {
    cat(sprintf("  %s: %s\n", k, paste(dim(ref[[k]]), collapse = "x")))
}

py_mel <- ref$mel
py_audio <- ref$audio_wav
py_speaker <- ref$speaker_embedding

cat(sprintf("\nPython mel stats:\n"))
cat(sprintf("  Shape: %s\n", paste(dim(py_mel), collapse = "x")))
cat(sprintf("  Mean: %.6f\n", mean(as.numeric(py_mel))))
cat(sprintf("  Std: %.6f\n", sd(as.numeric(py_mel))))
cat(sprintf("  Min: %.6f\n", min(as.numeric(py_mel))))
cat(sprintf("  Max: %.6f\n", max(as.numeric(py_mel))))

cat("\n=== Computing R mel ===\n")

# Python VE params (from hp):
# n_fft: 400, hop_size: 160, num_mels: 40
# sample_rate: 16000, fmin: 0, fmax: 8000
# mel_type: amp, mel_power: 2.0

# Convert audio to tensor
audio <- torch::torch_tensor(as.numeric(py_audio))$unsqueeze(1) # (1, samples)
cat(sprintf("Audio shape: %s\n", paste(dim(audio), collapse = "x")))

# Use the VE config that matches Python
config <- voice_encoder_config()
cat(sprintf("VE config: n_fft=%d, hop=%d, n_mels=%d, sr=%d\n",
        config$n_fft, config$hop_size, config$num_mels, config$sample_rate))

# Compute mel using R's voice encoder function
cat("Using compute_ve_mel from R/voice_encoder.R\n")
r_mel <- compute_ve_mel(audio, config)
# r_mel is (batch, time, mels) - transpose to (mels, time) for comparison
r_mel <- r_mel$squeeze(1)$transpose(1, 2) # Now (mels, time)

cat(sprintf("\nR mel stats:\n"))
cat(sprintf("  Shape: %s\n", paste(dim(r_mel), collapse = "x")))
cat(sprintf("  Mean: %.6f\n", r_mel$mean()$item()))
cat(sprintf("  Std: %.6f\n", r_mel$std()$item()))
cat(sprintf("  Min: %.6f\n", r_mel$min()$item()))
cat(sprintf("  Max: %.6f\n", r_mel$max()$item()))

cat("\n=== Comparison ===\n")

# Convert Python mel to tensor for comparison
py_mel_t <- torch::torch_tensor(as.numeric(py_mel))$view(c(dim(py_mel)[1], dim(py_mel)[2]))

# Make sure shapes match
r_mel_squeezed <- r_mel$squeeze()
if (length(dim(r_mel_squeezed)) == 2) {
    # R mel might be (n_mels, time) or (time, n_mels)
    cat(sprintf("R mel squeezed shape: %s\n", paste(dim(r_mel_squeezed), collapse = "x")))
    cat(sprintf("Python mel shape: %s\n", paste(dim(py_mel_t), collapse = "x")))

    # Check if we need to transpose
    if (dim(r_mel_squeezed)[1] != dim(py_mel_t)[1]) {
        cat("Transposing R mel to match Python...\n")
        r_mel_squeezed <- r_mel_squeezed$t()
    }

    # Trim to same length if needed
    min_time <- min(dim(r_mel_squeezed)[2], dim(py_mel_t)[2])
    r_mel_cmp <- r_mel_squeezed[, 1:min_time]
    py_mel_cmp <- py_mel_t[, 1:min_time]

    diff <- (r_mel_cmp - py_mel_cmp)$abs()
    cat(sprintf("\nDifference stats (first %d frames):\n", min_time))
    cat(sprintf("  Max diff: %.6f\n", diff$max()$item()))
    cat(sprintf("  Mean diff: %.6f\n", diff$mean()$item()))
    cat(sprintf("  Relative error: %.4f%%\n",
            100 * diff$mean()$item() / (py_mel_cmp$abs()$mean()$item() + 1e-8)))

    # Check first few values
    cat("\nFirst 5 values comparison (mel[1, 1:5]):\n")
    cat(sprintf("  Python: %s\n", paste(sprintf("%.4f", as.numeric(py_mel_cmp[1, 1:5])), collapse = ", ")))
    cat(sprintf("  R:      %s\n", paste(sprintf("%.4f", as.numeric(r_mel_cmp[1, 1:5])), collapse = ", ")))
} else {
    cat(sprintf("Unexpected R mel dimensions: %d\n", length(dim(r_mel_squeezed))))
}

cat("\nDone.\n")

