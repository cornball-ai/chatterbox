#!/usr/bin/env r
# Compare mel filterbank matrices between Python and R

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")

cat("=== Loading Python filterbank reference ===\n")
ref <- read_safetensors("~/chatterbox/outputs/filterbank_reference.safetensors")

py_fb <- torch::torch_tensor(as.numeric(ref$mel_filterbank))$view(c(40, 201))
py_fft_freqs <- as.numeric(ref$fft_freqs)
py_mel_freqs <- as.numeric(ref$mel_freqs)

cat(sprintf("Python filterbank shape: %s\n", paste(dim(py_fb), collapse = "x")))
cat(sprintf("Python FFT freqs (first 10): %s\n",
        paste(sprintf("%.1f", py_fft_freqs[1:10]), collapse = ", ")))
cat(sprintf("Python mel center freqs (first 10): %s\n",
        paste(sprintf("%.1f", py_mel_freqs[1:10]), collapse = ", ")))

cat("\n=== Creating R filterbank ===\n")

# Use same parameters as Python
r_fb <- create_mel_filterbank(
    sr = 16000,
    n_fft = 400,
    n_mels = 40,
    fmin = 0,
    fmax = 8000,
    norm = "slaney"
)

cat(sprintf("R filterbank shape: %dx%d\n", nrow(r_fb), ncol(r_fb)))

# Convert to tensor for comparison
r_fb_t <- torch::torch_tensor(r_fb, dtype = torch::torch_float32())

cat("\n=== Comparing filterbanks ===\n")

# Check if shapes match
if (all(dim(py_fb) == dim(r_fb_t))) {
    diff <- (py_fb - r_fb_t)$abs()
    cat(sprintf("Max diff: %.6f\n", diff$max()$item()))
    cat(sprintf("Mean diff: %.6f\n", diff$mean()$item()))

    # Check individual mel bins
    cat("\nPer-bin comparison:\n")
    for (bin in c(1, 10, 20, 30, 40)) {
        py_row <- as.numeric(py_fb[bin,])
        r_row <- as.numeric(r_fb_t[bin,])
        row_diff <- abs(py_row - r_row)

        cat(sprintf("  Bin %2d: py_sum=%.4f, r_sum=%.4f, max_diff=%.6f\n",
                bin, sum(py_row), sum(r_row), max(row_diff)))
    }
} else {
    cat("Shape mismatch!\n")
    cat(sprintf("Python: %s\n", paste(dim(py_fb), collapse = "x")))
    cat(sprintf("R: %s\n", paste(dim(r_fb_t), collapse = "x")))
}

# Compare mel frequencies
cat("\n=== R mel frequencies ===\n")
hz_to_mel <- function (hz) 2595 * log10(1 + hz / 700)
mel_to_hz <- function(mel) 700 * (10 ^ (mel / 2595) - 1)

mel_min <- hz_to_mel(0)
mel_max <- hz_to_mel(8000)
r_mel_points <- seq(mel_min, mel_max, length.out = 42) # n_mels + 2
r_hz_points <- mel_to_hz(r_mel_points)

cat(sprintf("R mel center freqs (first 10): %s\n",
        paste(sprintf("%.1f", r_hz_points[1:10]), collapse = ", ")))

# Check if HTK vs Slaney formula matters
cat("\n=== Checking formula differences ===\n")
# Slaney/librosa uses different formula below 1000 Hz
slaney_to_mel <- function(hz) {
    # Slaney formula: linear below f_sp=200/3 Hz, then log scale
    f_sp <- 200 / 3
    min_log_hz <- 1000
    logstep <- 0.06875177742094912# log(6.4)/27

    ifelse(hz < min_log_hz,
        (hz - 0) / f_sp,
        15 + log(hz / min_log_hz) / logstep)
}

cat(sprintf("HTK mel(1000Hz): %.1f\n", hz_to_mel(1000)))
cat(sprintf("Slaney mel(1000Hz): %.1f\n", slaney_to_mel(1000)))

cat("\nDone.\n")

