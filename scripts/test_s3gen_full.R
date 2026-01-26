#!/usr/bin/env Rscript
# Test full S3Gen pipeline including vocoder

library(chatterbox)

cat("=== S3Gen Full Pipeline Test ===\n\n")

# Load the full S3Gen model
cat("Loading S3Gen model...\n")
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")

if (!file.exists(weights_path)) {
    stop("S3Gen weights not found. Run load_chatterbox() first to download.")
}

model <- chatterbox:::load_s3gen(weights_path, device = "cpu")
model$eval()

cat("  Tokenizer: loaded\n")
cat("  Speaker encoder: loaded\n")
cat("  Flow: loaded\n")
cat("  Vocoder: loaded\n")

# Load reference audio
cat("\nLoading reference audio...\n")
ref_file <- "scripts/reference.wav"
if (!file.exists(ref_file)) {
    stop("Reference audio not found: ", ref_file)
}

ref_audio <- tuneR::readWave(ref_file)
ref_wav <- as.numeric(ref_audio@left) / 32768# Normalize to [-1, 1]
ref_sr <- ref_audio@samp.rate

cat(sprintf("  Reference: %d samples at %d Hz (%.2f sec)\n",
        length(ref_wav), ref_sr, length(ref_wav) / ref_sr))

# Embed reference
cat("\nEmbedding reference audio...\n")
ref_dict <- model$embed_ref(ref_wav, ref_sr)

cat(sprintf("  prompt_token: %s\n", paste(dim(ref_dict$prompt_token), collapse = "x")))
cat(sprintf("  prompt_feat: %s\n", paste(dim(ref_dict$prompt_feat), collapse = "x")))
cat(sprintf("  embedding: %s\n", paste(dim(ref_dict$embedding), collapse = "x")))

# Create test speech tokens (simulating T3 output)
cat("\nGenerating test speech tokens...\n")
set.seed(42)
# Create 50 tokens (about 2 seconds of speech at 25 tokens/sec)
test_tokens <- torch::torch_randint(low = 0L, high = 6560L, size = c(1L, 50L))
cat(sprintf("  Test tokens: %s\n", paste(dim(test_tokens), collapse = "x")))

# Run inference
cat("\nRunning S3Gen inference (flow + vocoder)...\n")
torch::with_no_grad({
        result <- model$inference(
            speech_tokens = test_tokens,
            ref_dict = ref_dict,
            finalize = TRUE
        )
    })

output_audio <- result[[1]]

cat(sprintf("\n=== Output ===\n"))
cat(sprintf("  Shape: %s\n", paste(dim(output_audio), collapse = "x")))
cat(sprintf("  Mean: %.6f\n", output_audio$mean()$item()))
cat(sprintf("  Std: %.6f\n", output_audio$std()$item()))
cat(sprintf("  Range: [%.4f, %.4f]\n",
        output_audio$min()$item(), output_audio$max()$item()))

# Calculate duration
n_samples <- output_audio$size(2)
duration <- n_samples / 24000
cat(sprintf("  Duration: %.2f seconds (%d samples at 24kHz)\n", duration, n_samples))

# Save output
output_file <- "outputs/s3gen_test_r.wav"
tryCatch({
        audio_vec <- as.numeric(output_audio$squeeze()$cpu())
        audio_int <- as.integer(audio_vec * 32767)
        wave_obj <- tuneR::Wave(left = audio_int, samp.rate = 24000, bit = 16)
        tuneR::writeWave(wave_obj, output_file)
        cat(sprintf("  Saved to: %s\n", output_file))
    }, error = function (e)
    {
        cat("  Could not save audio:", e$message, "\n")
    })

cat("\n=== S3Gen Full Pipeline Test Complete ===\n")

