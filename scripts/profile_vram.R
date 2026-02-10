#!/usr/bin/env r
# VRAM profile: step-by-step memory usage for native chatterbox

library(torch)

vram <- function(label) {
    smi <- system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", intern = TRUE)
    smi_mb <- as.numeric(trimws(smi[1]))
    cat(sprintf("%-45s  %5.0f MB\n", label, smi_mb))
    invisible(smi_mb)
}

cat("=== VRAM Profile: Native Chatterbox ===\n\n")
vram("Baseline (torch loaded)")

library(chatterbox)
vram("After library(chatterbox)")

# Step 1: Create model (no weights)
model <- chatterbox("cuda")
vram("After chatterbox('cuda') [empty model]")

# Step 2: Load weights
model <- load_chatterbox(model)
gc()
vram("After load_chatterbox + gc()")

# Step 3: Voice embedding
ref_audio <- system.file("audio", "jfk.mp3", package = "chatterbox")
voice <- create_voice_embedding(model, ref_audio)
gc()
vram("After voice embedding + gc()")

# Step 4: Generate (R backend)
cat("\n--- R backend generation ---\n")
result_r <- generate(model, "The quick brown fox jumps over the lazy dog.", voice, backend = "r")
vram("After R generate()")
rm(result_r); gc()
vram("After rm(result_r) + gc()")

# Step 5: Generate (C++ backend)
cat("\n--- C++ backend generation ---\n")
result_cpp <- generate(model, "The quick brown fox jumps over the lazy dog.", voice, backend = "cpp")
vram("After C++ generate()")
rm(result_cpp); gc()
vram("After rm(result_cpp) + gc()")

# Step 6: Second generation to check accumulation
cat("\n--- Second generation (leak check) ---\n")
result2 <- generate(model, "This is a second sentence to check for leaks.", voice, backend = "r")
vram("After 2nd R generate()")
rm(result2); gc()
vram("After rm + gc()")

result3 <- generate(model, "And a third one.", voice, backend = "cpp")
vram("After 3rd C++ generate()")
rm(result3); gc()
vram("After rm + gc()")

# Cleanup
cat("\n--- Cleanup ---\n")
rm(model, voice)
gc()
vram("After rm(model, voice) + gc()")
