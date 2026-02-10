#!/usr/bin/env r
# VRAM profile using cuda_memory_stats() + cuda_empty_cache()
# Distinguishes allocator cache from actual tensor retention

library(torch)
library(chatterbox)

vram <- function(label) {
    cuda_synchronize()
    stats <- cuda_memory_stats()
    alloc <- stats$allocated_bytes$all$current / 1024^2
    reserved <- stats$reserved_bytes$all$current / 1024^2
    smi <- as.numeric(trimws(system(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
        intern = TRUE
    )[1]))
    cat(sprintf("%-50s  alloc=%7.1f  reserved=%7.1f  smi=%5.0f MB\n",
        label, alloc, reserved, smi))
}

device <- "cuda"
ref_audio <- system.file("audio", "jfk.mp3", package = "chatterbox")
text <- "The quick brown fox jumps over the lazy dog."

cat("=== VRAM Profile (cuda_memory_stats) ===\n\n")
vram("Baseline")

# Load model
model <- chatterbox(device)
model <- load_chatterbox(model)
gc(); cuda_empty_cache()
vram("Model loaded + empty_cache")

# Voice embedding
voice <- create_voice_embedding(model, ref_audio)
gc(); cuda_empty_cache()
vram("Voice embedding + empty_cache")

# --- Generation 1 (R backend) ---
cat("\n--- Generation 1 (R backend) ---\n")
result1 <- generate(model, text, voice, backend = "r")
vram("After generate (before cleanup)")

rm(result1); gc()
vram("After rm + gc (no empty_cache)")

cuda_empty_cache()
vram("After cuda_empty_cache")

# --- Generation 2 (accumulation check) ---
cat("\n--- Generation 2 (leak check) ---\n")
result2 <- generate(model, "This is a second sentence to check for leaks.", voice, backend = "r")
vram("After 2nd generate")

rm(result2); gc(); cuda_empty_cache()
vram("After 2nd rm + gc + empty_cache")

# --- Generation 3 ---
cat("\n--- Generation 3 (C++ backend) ---\n")
result3 <- generate(model, text, voice, backend = "cpp")
vram("After C++ generate")

rm(result3); gc(); cuda_empty_cache()
vram("After C++ rm + gc + empty_cache")

# --- Cleanup ---
cat("\n--- Cleanup ---\n")
rm(model, voice)
gc(); cuda_empty_cache()
vram("After rm(model, voice) + empty_cache")
