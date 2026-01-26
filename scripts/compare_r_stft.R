#!/usr/bin/env r
# Compare R STFT against Python reference

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")
source("~/chatterbox/R/voice_encoder.R")

# Load Python reference
ref <- read_safetensors("~/chatterbox/outputs/mel_reference.safetensors")
py_audio <- ref$audio_wav

# Load audio as tensor
audio <- torch::torch_tensor(as.numeric(py_audio))$unsqueeze(1)
cat(sprintf("Audio shape: %s, samples: %d\n", paste(dim(audio), collapse = "x"), audio$size(2)))

# STFT parameters (matching Python)
n_fft <- 400L
hop_size <- 160L
win_size <- 400L

# Create Hann window
hann_window <- torch::torch_hann_window(win_size)

# Method 1: center=TRUE (should match Python)
stft_center <- torch::torch_stft(
    audio,
    n_fft = n_fft,
    hop_length = hop_size,
    win_length = win_size,
    window = hann_window,
    center = TRUE,
    pad_mode = "reflect",
    normalized = FALSE,
    onesided = TRUE,
    return_complex = TRUE
)

# Get magnitude
mag_center <- torch::torch_abs(stft_center)$squeeze(1)

cat(sprintf("\nR STFT (center=TRUE):\n"))
cat(sprintf("  Shape: %s\n", paste(dim(mag_center), collapse = "x")))
cat(sprintf("  Magnitude mean: %.6f\n", mag_center$mean()$item()))
cat(sprintf("  Magnitude max: %.6f\n", mag_center$max()$item()))

# Method 2: Manual padding (what compute_ve_mel does)
pad_amount <- n_fft %/% 2
audio_padded <- audio$unsqueeze(2)
audio_padded <- torch::nnf_pad(audio_padded, c(pad_amount, pad_amount), mode = "reflect")
audio_padded <- audio_padded$squeeze(2)

stft_manual <- torch::torch_stft(
    audio_padded,
    n_fft = n_fft,
    hop_length = hop_size,
    win_length = win_size,
    window = hann_window,
    center = FALSE,
    pad_mode = "reflect",
    normalized = FALSE,
    onesided = TRUE,
    return_complex = TRUE
)

mag_manual <- torch::torch_abs(stft_manual)$squeeze(1)

cat(sprintf("\nR STFT (center=FALSE, manual padding):\n"))
cat(sprintf("  Shape: %s\n", paste(dim(mag_manual), collapse = "x")))
cat(sprintf("  Magnitude mean: %.6f\n", mag_manual$mean()$item()))
cat(sprintf("  Magnitude max: %.6f\n", mag_manual$max()$item()))

# Compare the two
diff <- (mag_center - mag_manual[, 1:dim(mag_center)[2]])$abs()
cat(sprintf("\nDiff between center=TRUE and manual: max=%.6f, mean=%.6f\n",
        diff$max()$item(), diff$mean()$item()))

# Frame 100 comparison
cat(sprintf("\nFrame 100 (first 10 freq bins):\n"))
cat(sprintf("  R center:  %s\n", paste(sprintf("%.4f", as.numeric(mag_center[1:10, 100])), collapse = ", ")))
cat(sprintf("  R manual:  %s\n", paste(sprintf("%.4f", as.numeric(mag_manual[1:10, 100])), collapse = ", ")))

# Python reference values from earlier run
cat(sprintf("  Py (ref):  0.0876, 0.1110, 0.2919, 0.5119, 0.1598, 0.3209, 0.8281, 1.0784, 0.4031, 0.9393\n"))

cat("\nDone.\n")

