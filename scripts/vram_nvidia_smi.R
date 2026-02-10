#!/usr/bin/env r
# Compare cuda_memory_stats() vs nvidia-smi

library(torch)

cuda_mem <- function(label = "") {
    allocated <- cuda_memory_stats()$allocated_bytes$all$current / 1024^3
    reserved <- cuda_memory_stats()$reserved_bytes$all$current / 1024^3

    # Get nvidia-smi reading
    smi <- system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", intern = TRUE)
    smi_mb <- as.numeric(trimws(smi[1]))

    cat(sprintf("%s:\n", label))
    cat(sprintf("  CUDA allocated: %.2f GB\n", allocated))
    cat(sprintf("  CUDA reserved:  %.2f GB\n", reserved))
    cat(sprintf("  nvidia-smi:     %.2f GB\n", smi_mb / 1024))
    cat("\n")
}

cat("=== CUDA stats vs nvidia-smi ===\n\n")

cuda_mem("Initial (no torch)")

rhydrogen::load_all("/home/troy/chatterbox")

device <- "cuda"
model <- chatterbox(device)
model <- load_chatterbox(model)

cuda_mem("After model load")

gc()
cuda_empty_cache()
cuda_mem("After cleanup")

# Run TTS
ref_audio <- "/home/troy/Music/cornball_jfk.wav"
voice <- create_voice_embedding(model, ref_audio)
result <- generate(model, "Hello, this is a test.", voice)

cuda_mem("After TTS (before cleanup)")

gc()
cuda_empty_cache()
cuda_mem("After TTS (after cleanup)")
