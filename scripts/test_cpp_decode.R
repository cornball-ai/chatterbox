#!/usr/bin/env r
# Test C++ T3 decode loop
#
# Verifies that the C++ backend produces valid speech tokens and audio,
# and benchmarks against the R backend.

library(chatterbox)
library(torch)

device <- if (cuda_is_available()) "cuda" else "cpu"
cat(sprintf("Using device: %s\n", device))

# Load model
cat("Loading model...\n")
model <- chatterbox(device)
model <- load_chatterbox(model)

# Create voice embedding
cat("Creating voice embedding...\n")
voice <- create_voice_embedding(model, "inst/audio/jfk.mp3")

text <- "The quick brown fox jumps over the lazy dog."

# ============================================================================
# Test 1: C++ backend produces valid tokens
# ============================================================================
cat("\n=== Test 1: C++ backend produces valid tokens ===\n")

result_cpp <- generate(model, text, voice, backend = "cpp")
n_samples <- length(result_cpp$audio)
duration <- n_samples / result_cpp$sample_rate

cat(sprintf("Audio samples: %d\n", n_samples))
cat(sprintf("Duration: %.2f seconds\n", duration))
cat(sprintf("Sample rate: %d Hz\n", result_cpp$sample_rate))

stopifnot(n_samples > 0)
stopifnot(duration > 0.5)  # Should be at least half a second
stopifnot(result_cpp$sample_rate == 24000)
cat("PASS: C++ backend produces valid audio\n")

# ============================================================================
# Test 2: Audio is not silence or noise
# ============================================================================
cat("\n=== Test 2: Audio quality check ===\n")

audio <- result_cpp$audio
rms <- sqrt(mean(audio^2))
peak <- max(abs(audio))

cat(sprintf("RMS: %.4f\n", rms))
cat(sprintf("Peak: %.4f\n", peak))

stopifnot(rms > 0.01)   # Not silence
stopifnot(peak < 2.0)   # Not clipping badly
stopifnot(peak > 0.05)  # Has actual content
cat("PASS: Audio has reasonable signal levels\n")

# ============================================================================
# Test 3: Save and verify output
# ============================================================================
cat("\n=== Test 3: Save audio ===\n")

outpath <- tempfile(fileext = ".wav")
write_audio(result_cpp$audio, result_cpp$sample_rate, outpath)
stopifnot(file.exists(outpath))
cat(sprintf("PASS: Audio saved to %s\n", outpath))

# ============================================================================
# Test 4: Benchmark R vs C++
# ============================================================================
cat("\n=== Test 4: Benchmark R vs C++ ===\n")

# Warm up
generate(model, "Hello", voice, backend = "r")
generate(model, "Hello", voice, backend = "cpp")

n_runs <- 3
r_times <- numeric(n_runs)
cpp_times <- numeric(n_runs)

for (i in seq_len(n_runs)) {
    t_r <- system.time(generate(model, text, voice, backend = "r"))
    r_times[i] <- t_r[3]

    t_cpp <- system.time(generate(model, text, voice, backend = "cpp"))
    cpp_times[i] <- t_cpp[3]
}

cat(sprintf("\nR backend:   %.2fs avg (%.2f, %.2f, %.2f)\n",
    mean(r_times), r_times[1], r_times[2], r_times[3]))
cat(sprintf("C++ backend: %.2fs avg (%.2f, %.2f, %.2f)\n",
    mean(cpp_times), cpp_times[1], cpp_times[2], cpp_times[3]))
cat(sprintf("Speedup: %.1fx\n", mean(r_times) / mean(cpp_times)))

stopifnot(mean(cpp_times) < mean(r_times))  # C++ should be faster
cat("PASS: C++ backend is faster than R\n")

cat("\nAll tests passed.\n")
