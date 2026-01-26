#!/usr/bin/env r
# Profile VRAM usage during chatterbox inference

library(torch)

cuda_mem <- function(label = "") {
    if (cuda_is_available()) {
        allocated <- cuda_memory_stats()$allocated_bytes$all$current / 1024^3
        reserved <- cuda_memory_stats()$reserved_bytes$all$current / 1024^3
        cat(sprintf("%s: %.2f GB allocated, %.2f GB reserved\n",
                    label, allocated, reserved))
        invisible(allocated)
    }
}

cat("=== VRAM Profile for Chatterbox Inference ===\n\n")

rhydrogen::load_all("/home/troy/chatterbox")

device <- "cuda"
model <- chatterbox(device)
model <- load_chatterbox(model)

gc()
cuda_empty_cache()
cuda_mem("After model load + cleanup")

# Create voice embedding
cat("\n--- Voice Embedding ---\n")
ref_audio <- "/home/troy/Music/cornball_jfk.wav"
voice <- create_voice_embedding(model, ref_audio)
cuda_mem("After voice embedding")

gc()
cuda_empty_cache()
cuda_mem("After voice embedding + cleanup")

# Run TTS
cat("\n--- TTS Inference ---\n")
text <- "Hello, this is a test of the text to speech system."

result <- tts(model, text, voice)

cuda_mem("After TTS")

gc()
cuda_empty_cache()
cuda_mem("After TTS + cleanup")

# Second TTS call - does memory accumulate?
cat("\n--- Second TTS Inference ---\n")
result2 <- tts(model, "This is a second test to check for memory accumulation.", voice)
cuda_mem("After 2nd TTS")

gc()
cuda_empty_cache()
cuda_mem("After 2nd TTS + cleanup")

# Third TTS call
cat("\n--- Third TTS Inference ---\n")
result3 <- tts(model, "And a third test just to be sure.", voice)
cuda_mem("After 3rd TTS")

gc()
cuda_empty_cache()
cuda_mem("After 3rd TTS + cleanup")

cat("\n--- What's still allocated? ---\n")
cat("Model components are still in memory.\n")
cat("Voice embedding tensors are still in memory.\n")

# Check voice embedding sizes
cat("\nVoice embedding contents:\n")
cat(sprintf("  ve_embedding: %s (%.1f MB)\n",
            paste(dim(voice$ve_embedding), collapse="x"),
            prod(dim(voice$ve_embedding)) * 4 / 1024^2))
cat(sprintf("  cond_prompt_speech_tokens: %s\n",
            paste(dim(voice$cond_prompt_speech_tokens), collapse="x")))

# Check ref_dict
cat("  ref_dict keys:", paste(names(voice$ref_dict), collapse=", "), "\n")
for (k in names(voice$ref_dict)) {
    v <- voice$ref_dict[[k]]
    if (inherits(v, "torch_tensor")) {
        cat(sprintf("    %s: %s (%s) %.1f MB\n",
                    k, paste(dim(v), collapse="x"), v$dtype$.type(),
                    prod(dim(v)) * 4 / 1024^2))
    }
}
