#!/usr/bin/env r
# Trace mel computation step by step

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")

# Load audio
ref <- read_safetensors("~/chatterbox/outputs/mel_reference.safetensors")
py_audio <- ref$audio_wav
audio <- torch::torch_tensor(as.numeric(py_audio))$unsqueeze(1)

cat(sprintf("Audio: %d samples\n", audio$size(2)))

# Parameters (matching Python VE)
n_fft <- 400L
hop_size <- 160L
win_size <- 400L
n_mels <- 40L
sr <- 16000L
fmin <- 0
fmax <- 8000

# Step 1: Create window
hann_window <- torch::torch_hann_window(win_size)
cat(sprintf("\n1. Window sum: %.6f (expected: 200 for Hann)\n", hann_window$sum()$item()))

# Step 2: STFT with center=TRUE (matching Python librosa)
stft_result <- torch::torch_stft(
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
cat(sprintf("\n2. STFT shape: %s\n", paste(dim(stft_result), collapse = "x")))

# Step 3: Get magnitude
magnitude <- torch::torch_abs(stft_result)$squeeze(1)
cat(sprintf("\n3. Magnitude: mean=%.6f, max=%.6f\n", magnitude$mean()$item(), magnitude$max()$item()))
cat(sprintf("   Frame 101 (first 5): %s\n",
        paste(sprintf("%.4f", as.numeric(magnitude[1:5, 101])), collapse = ", ")))

# Step 4: Apply power (mel_power = 2.0)
mag_pow <- magnitude$pow(2)
cat(sprintf("\n4. After power(2): mean=%.6f, max=%.6f\n", mag_pow$mean()$item(), mag_pow$max()$item()))
cat(sprintf("   Frame 101 (first 5): %s\n",
        paste(sprintf("%.4f", as.numeric(mag_pow[1:5, 101])), collapse = ", ")))

# Step 5: Create mel filterbank
mel_fb <- create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
mel_fb_t <- torch::torch_tensor(mel_fb, dtype = torch::torch_float32())
cat(sprintf("\n5. Mel filterbank: %dx%d, sum=%.6f\n", nrow(mel_fb), ncol(mel_fb), sum(mel_fb)))

# Step 6: Apply filterbank (matmul)
# mel_fb is (n_mels, n_fft_bins) = (40, 201)
# mag_pow is (n_fft_bins, time) = (201, 754)
# result should be (n_mels, time) = (40, 754)
mel <- torch::torch_matmul(mel_fb_t, mag_pow)
cat(sprintf("\n6. Final mel: shape=%s, mean=%.6f, max=%.6f\n",
        paste(dim(mel), collapse = "x"), mel$mean()$item(), mel$max()$item()))
cat(sprintf("   Frame 101 (bins 15-25): %s\n",
        paste(sprintf("%.4f", as.numeric(mel[15:25, 101])), collapse = ", ")))

# Compare with Python reference
py_mel <- torch::torch_tensor(as.numeric(ref$mel))$view(c(40, 754))
cat(sprintf("\nPython mel: mean=%.6f, max=%.6f\n", py_mel$mean()$item(), py_mel$max()$item()))
# Python frame 100 (0-indexed) = our frame 101 (1-indexed for tensor but we saved 0-indexed)
# Actually when loaded from safetensors, Python 0-indexed becomes R 1-indexed
cat(sprintf("   Python frame 101 (bins 15-25): %s\n",
        paste(sprintf("%.4f", as.numeric(py_mel[15:25, 101])), collapse = ", ")))

# Try frame 100
cat(sprintf("   Python frame 100 (bins 15-25): %s\n",
        paste(sprintf("%.4f", as.numeric(py_mel[15:25, 100])), collapse = ", ")))

# Difference
min_len <- min(dim(mel)[2], dim(py_mel)[2])
diff <- (mel[, 1:min_len] - py_mel[, 1:min_len])$abs()
cat(sprintf("\nMel diff: mean=%.6f, max=%.6f\n", diff$mean()$item(), diff$max()$item()))

cat("\nDone.\n")

