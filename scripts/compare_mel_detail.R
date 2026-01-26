#!/usr/bin/env r
# Detailed mel spectrogram comparison

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")
source("~/chatterbox/R/voice_encoder.R")

cat("=== Loading Python reference ===\n")
ref <- read_safetensors("~/chatterbox/outputs/mel_reference.safetensors")

py_mel <- torch::torch_tensor(as.numeric(ref$mel))$view(c(40, 754))
py_audio <- ref$audio_wav

cat(sprintf("Python mel shape: %s\n", paste(dim(py_mel), collapse = "x")))

# Compute R mel
audio <- torch::torch_tensor(as.numeric(py_audio))$unsqueeze(1)
config <- voice_encoder_config()
r_mel <- compute_ve_mel(audio, config)$squeeze(1)$transpose(1, 2)

cat(sprintf("R mel shape: %s\n", paste(dim(r_mel), collapse = "x")))

# Compare at different mel bins
cat("\n=== Comparing mel bins ===\n")
min_time <- min(dim(r_mel)[2], dim(py_mel)[2])

for (mel_bin in c(1, 10, 20, 30, 40)) {
    py_row <- py_mel[mel_bin, 1:min_time]
    r_row <- r_mel[mel_bin, 1:min_time]

    diff <- (py_row - r_row)$abs()

    cat(sprintf("\nMel bin %d:\n", mel_bin))
    cat(sprintf("  Python: mean=%.4f, max=%.4f\n", py_row$mean()$item(), py_row$max()$item()))
    cat(sprintf("  R:      mean=%.4f, max=%.4f\n", r_row$mean()$item(), r_row$max()$item()))
    cat(sprintf("  Diff:   mean=%.4f, max=%.4f\n", diff$mean()$item(), diff$max()$item()))

    # Find where max difference occurs
    max_idx <- as.integer(diff$argmax()$item()) + 1
    cat(sprintf("  Max diff at frame %d: py=%.4f, r=%.4f\n",
            max_idx, py_row[max_idx]$item(), r_row[max_idx]$item()))
}

# Look at a specific time frame
cat("\n=== Frame 100 comparison (all 40 mels) ===\n")
frame <- 100
py_frame <- as.numeric(py_mel[, frame])
r_frame <- as.numeric(r_mel[, frame])

cat("Mel bin | Python | R      | Diff\n")
cat("--------|--------|--------|------\n")
for (i in seq_along(py_frame)) {
    diff_val <- abs(py_frame[i] - r_frame[i])
    if (diff_val > 0.01) {
        cat(sprintf("%-7d | %.4f | %.4f | %.4f *\n", i, py_frame[i], r_frame[i], diff_val))
    }
}

# Check if it's an offset issue
cat("\n=== Checking for frame offset ===\n")
# Compute correlation at different offsets
for (offset in - 3:3) {
    if (offset < 0) {
        py_slice <- py_mel[, 1:(min_time + offset)]
        r_slice <- r_mel[, (1 - offset) :min_time]
    } else if (offset > 0) {
        py_slice <- py_mel[, (1 + offset) :min_time]
        r_slice <- r_mel[, 1:(min_time - offset)]
    } else {
        py_slice <- py_mel[, 1:min_time]
        r_slice <- r_mel[, 1:min_time]
    }

    diff <- (py_slice - r_slice)$abs()$mean()$item()
    cat(sprintf("  Offset %+d: mean diff = %.6f\n", offset, diff))
}

cat("\nDone.\n")

