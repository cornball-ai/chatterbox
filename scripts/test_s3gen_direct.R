#!/usr/bin/env Rscript
# Test S3Gen directly with Python reference tokens to bypass slow T3

library(chatterbox)

cat("=== S3Gen Direct Test ===\n\n")

# Load S3Gen model
cat("Loading S3Gen...\n")
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
model <- chatterbox:::load_s3gen(weights_path, device = "cpu")
model$eval()

# Load reference audio for embedding
cat("\nEmbedding reference audio...\n")
ref_audio <- tuneR::readWave("scripts/reference.wav")
ref_wav <- as.numeric(ref_audio@left) / 32768
ref_sr <- ref_audio@samp.rate

# Get reference embedding
ref_dict <- model$embed_ref(ref_wav, ref_sr)
cat(sprintf("  prompt_token: %s\n", paste(dim(ref_dict$prompt_token), collapse = "x")))
cat(sprintf("  embedding: %s\n", paste(dim(ref_dict$embedding), collapse = "x")))

# Create simple test tokens (valid range 0-6560)
cat("\nCreating test tokens (50 tokens, range 0-6560)...\n")
set.seed(42)
test_tokens <- torch::torch_tensor(
    matrix(sample(0:6560, 50, replace = TRUE), nrow = 1),
    dtype = torch::torch_long()
)
cat(sprintf("  Tokens shape: %s\n", paste(dim(test_tokens), collapse = "x")))
cat(sprintf("  Token range: [%d, %d]\n",
        as.integer(test_tokens$min()$item()),
        as.integer(test_tokens$max()$item())))

# Run S3Gen inference
cat("\nRunning S3Gen inference...\n")
torch::with_no_grad({
        result <- model$inference(
            speech_tokens = test_tokens,
            ref_dict = ref_dict,
            finalize = TRUE
        )
    })

audio <- result[[1]]
cat(sprintf("\n=== Output ===\n"))
cat(sprintf("  Shape: %s\n", paste(dim(audio), collapse = "x")))
cat(sprintf("  Mean: %.6f, Std: %.6f\n",
        audio$mean()$item(), audio$std()$item()))
cat(sprintf("  Range: [%.4f, %.4f]\n",
        audio$min()$item(), audio$max()$item()))
cat(sprintf("  Duration: %.2f sec\n", audio$size(2) / 24000))

abs_mean <- audio$abs()$mean()$item()
if (abs_mean > 0.01) {
    cat("\nSUCCESS: S3Gen produces audio with signal\n")

    # Save output
    audio_vec <- as.numeric(audio$squeeze()$cpu())
    audio_int <- as.integer(audio_vec * 32767)
    wave <- tuneR::Wave(left = audio_int, samp.rate = 24000, bit = 16)
    tuneR::writeWave(wave, "outputs/s3gen_direct_test.wav")
    cat("Saved to outputs/s3gen_direct_test.wav\n")
} else {
    cat("\nFAILURE: S3Gen output is silent\n")
}

