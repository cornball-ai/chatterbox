#!/usr/bin/env r
# Test R voice encoder against Python reference

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/audio_utils.R")
source("~/chatterbox/R/voice_encoder.R")

cat("Loading Python reference...\n")
ve_steps <- read_safetensors("~/chatterbox/outputs/ve_steps.safetensors")

cat("Keys:", paste(names(ve_steps), collapse = ", "), "\n\n")

# Load audio from mel_reference
ref <- read_safetensors("~/chatterbox/outputs/mel_reference.safetensors")
audio <- torch::torch_tensor(as.numeric(ref$audio_wav))

cat(sprintf("Audio samples: %d\n", length(audio)))

# ============================================================================
# Step 1: Compute mel spectrogram (already validated)
# ============================================================================
config <- voice_encoder_config()
mel <- compute_ve_mel(audio, config)
cat(sprintf("\n1. R mel shape: %s (expected batch, time, mels)\n",
        paste(dim(mel), collapse = "x")))

py_mel <- ve_steps$mel_for_ve
cat(sprintf("   Python mel shape: %s\n", paste(dim(py_mel), collapse = "x")))

# Check mel values (Python is time x mels, R is 1 x time x mels)
r_mel_2d <- mel$squeeze(1) # Remove batch dim for comparison
diff <- (r_mel_2d - py_mel)$abs()
cat(sprintf("   Mel diff: max=%.6f, mean=%.6f\n", diff$max()$item(), diff$mean()$item()))

# ============================================================================
# Step 2: Check partials
# ============================================================================
py_partials <- ve_steps$partials
cat(sprintf("\n2. Python partials shape: %s\n", paste(dim(py_partials), collapse = "x")))

n_frames <- mel$size(2)
partial_frames <- config$ve_partial_frames
frame_step <- as.integer(round(partial_frames * (1 - 0.5))) # overlap=0.5
n_partials <- (n_frames - partial_frames + frame_step) %/% frame_step
if (n_partials == 0) n_partials <- 1

cat(sprintf("   R n_frames=%d, partial_frames=%d, frame_step=%d, n_partials=%d\n",
        n_frames, partial_frames, frame_step, n_partials))

# Extract first partial for comparison
r_partial_1 <- mel[1, 1:160,]# First 160 frames
py_partial_1 <- py_partials[1,,]# First partial

diff <- (r_partial_1 - py_partial_1)$abs()
cat(sprintf("   Partial 1 diff: max=%.6f, mean=%.6f\n", diff$max()$item(), diff$mean()$item()))

# ============================================================================
# Step 3: Load VE weights and test forward pass
# ============================================================================
cat("\n3. Testing LSTM forward pass...\n")

# Load weights (extracted from Docker container)
weights_path <- "~/chatterbox/outputs/voice_encoder_weights.safetensors"
if (!file.exists(weights_path)) {
    cat("   Voice encoder weights not found at:", weights_path, "\n")
    cat("   Skipping LSTM test\n")
} else {
    ve_weights <- read_safetensors(weights_path)
    cat(sprintf("   Loaded %d weight tensors\n", length(ve_weights)))

    # Create model and load weights
    ve_model <- voice_encoder(config)
    ve_model <- load_voice_encoder_weights(ve_model, ve_weights)
    ve_model$eval()

    # Forward pass on first partial
    torch::with_no_grad({
            # Stack partials for batch processing
            r_partials <- list()
            for (i in seq_len(n_partials)) {
                start_idx <- (i - 1) * frame_step + 1
                end_idx <- min(start_idx + partial_frames - 1, n_frames)

                if (end_idx - start_idx + 1 < partial_frames) {
                    partial <- torch::torch_zeros(c(partial_frames, config$num_mels))
                    actual_len <- end_idx - start_idx + 1
                    partial[1:actual_len,] <- mel[1, start_idx:end_idx,]
                } else {
                    partial <- mel[1, start_idx:end_idx,]
                }
                r_partials[[i]] <- partial$unsqueeze(1)
            }
            partials_t <- torch::torch_cat(r_partials, dim = 1)
            cat(sprintf("   R partials tensor shape: %s\n", paste(dim(partials_t), collapse = "x")))

            # LSTM forward
            lstm_result <- ve_model$lstm$forward(partials_t)
            lstm_out <- lstm_result[[1]]
            hidden <- lstm_result[[2]][[1]]# Hidden states

            cat(sprintf("   LSTM output shape: %s\n", paste(dim(lstm_out), collapse = "x")))
            cat(sprintf("   Hidden shape: %s\n", paste(dim(hidden), collapse = "x")))

            # Get final layer hidden (layer 3)
            final_hidden <- hidden[3,,]# (batch, hidden_size)
            cat(sprintf("   Final hidden shape: %s\n", paste(dim(final_hidden), collapse = "x")))

            # Compare with Python
            py_final_hidden <- ve_steps$final_hidden
            diff <- (final_hidden - py_final_hidden)$abs()
            cat(sprintf("   Final hidden diff: max=%.6f, mean=%.6f\n", diff$max()$item(), diff$mean()$item()))

            # Project to embedding
            raw_embeds <- ve_model$proj$forward(final_hidden)
            py_raw_embeds <- ve_steps$raw_embeds
            diff <- (raw_embeds - py_raw_embeds)$abs()
            cat(sprintf("   Raw embeds diff: max=%.6f, mean=%.6f\n", diff$max()$item(), diff$mean()$item()))

            # ReLU
            relu_embeds <- torch::nnf_relu(raw_embeds)
            py_relu_embeds <- ve_steps$relu_embeds
            diff <- (relu_embeds - py_relu_embeds)$abs()
            cat(sprintf("   ReLU embeds diff: max=%.6f, mean=%.6f\n", diff$max()$item(), diff$mean()$item()))

            # L2 normalize partials
            partial_embeds <- relu_embeds / torch::torch_norm(relu_embeds, dim = 2, keepdim = TRUE)
            py_partial_embeds <- ve_steps$partial_embeds
            diff <- (partial_embeds - py_partial_embeds)$abs()
            cat(sprintf("   Partial embeds diff: max=%.6f, mean=%.6f\n", diff$max()$item(), diff$mean()$item()))

            # Average and normalize
            mean_embed <- torch::torch_mean(partial_embeds, dim = 1, keepdim = TRUE)
            speaker_embed <- mean_embed / torch::torch_norm(mean_embed, dim = 2, keepdim = TRUE)

            py_speaker <- ve_steps$speaker_embedding
            diff <- (speaker_embed - py_speaker)$abs()
            cat(sprintf("\n   Final speaker embedding diff: max=%.6f, mean=%.6f\n",
                    diff$max()$item(), diff$mean()$item()))

            cat(sprintf("\n   R speaker (first 10): %s\n",
                    paste(sprintf("%.6f", as.numeric(speaker_embed[1, 1:10])), collapse = ", ")))
            cat(sprintf("   Py speaker (first 10): %s\n",
                    paste(sprintf("%.6f", as.numeric(py_speaker[1, 1:10])), collapse = ", ")))
        })
}

cat("\nDone.\n")

