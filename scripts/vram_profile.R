#!/usr/bin/env r
# Profile VRAM usage at each step of chatterbox model loading

library(torch)

# Helper to get CUDA memory stats
cuda_mem <- function(label = "") {
    if (cuda_is_available()) {
        allocated <- cuda_memory_stats()$allocated_bytes$all$current / 1024^3
        reserved <- cuda_memory_stats()$reserved_bytes$all$current / 1024^3
        cat(sprintf("%s: %.2f GB allocated, %.2f GB reserved\n", label, allocated, reserved))
        invisible(allocated)
    } else {
        cat(label, ": CUDA not available\n")
        invisible(0)
    }
}

# Count parameters in a module
count_params <- function(module) {
    params <- module$parameters
    total <- 0
    for (p in params) {
        total <- total + prod(p$shape)
    }
    total
}

cat("=== VRAM Profile for Chatterbox Loading ===\n\n")

cuda_mem("Initial")

# Source the package
rhydrogen::load_all("/home/troy/chatterbox")

cuda_mem("After load_all")

# Get model paths
paths <- download_chatterbox_models()
cat("\nModel files:\n")
for (name in names(paths)) {
    if (file.exists(paths[[name]])) {
        size <- file.info(paths[[name]])$size / 1024^2
        cat(sprintf("  %s: %.1f MB\n", name, size))
    }
}

device <- "cuda"
cat("\n=== Loading Components ===\n\n")

# Voice encoder
cat("--- Voice Encoder ---\n")
cuda_mem("Before VE weights")
ve_weights <- read_safetensors(paths$ve, device)
cuda_mem("After VE weights load")
cat(sprintf("VE weights count: %d tensors\n", length(ve_weights)))

ve_model <- voice_encoder()
cuda_mem("After VE model create")

ve_model <- load_voice_encoder_weights(ve_model, ve_weights)
cuda_mem("After VE weights assign")

ve_model$to(device = device)
cuda_mem("After VE to(cuda)")

ve_params <- count_params(ve_model)
cat(sprintf("VE parameters: %d (%.1f MB @ float32)\n", ve_params, ve_params * 4 / 1024^2))

# Clear weights dict
rm(ve_weights)
gc()
cuda_empty_cache()
cuda_mem("After VE cleanup")

# T3 model
cat("\n--- T3 Model ---\n")
cuda_mem("Before T3 weights")
t3_weights <- read_safetensors(paths$t3_cfg, device)
cuda_mem("After T3 weights load")
cat(sprintf("T3 weights count: %d tensors\n", length(t3_weights)))

t3_model_inst <- t3_model()
cuda_mem("After T3 model create")

t3_model_inst <- load_t3_weights(t3_model_inst, t3_weights)
cuda_mem("After T3 weights assign")

t3_model_inst$to(device = device)
cuda_mem("After T3 to(cuda)")

t3_params <- count_params(t3_model_inst)
cat(sprintf("T3 parameters: %d (%.1f MB @ float32)\n", t3_params, t3_params * 4 / 1024^2))

rm(t3_weights)
gc()
cuda_empty_cache()
cuda_mem("After T3 cleanup")

# S3Gen model
cat("\n--- S3Gen Model ---\n")
cuda_mem("Before S3Gen weights")
s3_weights <- read_safetensors(paths$s3gen, device)
cuda_mem("After S3Gen weights load")
cat(sprintf("S3Gen weights count: %d tensors\n", length(s3_weights)))

s3_model <- s3gen()
cuda_mem("After S3Gen model create")

s3_model$mel2wav <- create_s3gen_vocoder(device)
cuda_mem("After vocoder create")

voc_params <- count_params(s3_model$mel2wav)
cat(sprintf("Vocoder parameters: %d (%.1f MB @ float32)\n", voc_params, voc_params * 4 / 1024^2))

s3_model <- load_s3gen_weights(s3_model, s3_weights)
cuda_mem("After S3Gen weights assign")

s3_model$to(device = device)
cuda_mem("After S3Gen to(cuda)")

s3_params <- count_params(s3_model)
cat(sprintf("S3Gen parameters (total): %d (%.1f MB @ float32)\n", s3_params, s3_params * 4 / 1024^2))

rm(s3_weights)
gc()
cuda_empty_cache()
cuda_mem("After S3Gen cleanup")

cat("\n=== Summary ===\n")
total_params <- ve_params + t3_params + s3_params
cat(sprintf("Total parameters: %d (%.1f MB @ float32)\n", total_params, total_params * 4 / 1024^2))
cuda_mem("Final")
