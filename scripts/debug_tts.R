#!/usr/bin/env Rscript
# Debug TTS pipeline to find where silence originates

library(chatterbox)

cat("=== TTS Pipeline Debug ===\n\n")

# Load model
cat("Loading model...\n")
model <- chatterbox("cpu")
model <- load_chatterbox(model)

# Load reference audio
cat("\nLoading reference audio (jfk.wav)...\n")
ref_file <- "inst/audio/jfk.wav"
ref_audio <- tuneR::readWave(ref_file)
ref_wav <- as.numeric(ref_audio@left) / 32768
ref_sr <- ref_audio@samp.rate
cat(sprintf("  Reference: %d samples at %d Hz (%.2f sec)\n",
        length(ref_wav), ref_sr, length(ref_wav) / ref_sr))

# Create voice embedding
cat("\n=== Step 1: Create Voice Embedding ===\n")
voice <- chatterbox:::create_voice_embedding(model, ref_wav, ref_sr)
cat(sprintf("  ve_embedding: %s\n", paste(dim(voice$ve_embedding), collapse = "x")))
cat(sprintf("  cond_prompt_speech_tokens: %s\n", paste(dim(voice$cond_prompt_speech_tokens), collapse = "x")))
cat(sprintf("  ref_dict$embedding (xvector): %s\n", paste(dim(voice$ref_dict$embedding), collapse = "x")))
cat(sprintf("  ref_dict$prompt_token: %s\n", paste(dim(voice$ref_dict$prompt_token), collapse = "x")))

# Tokenize text
cat("\n=== Step 2: Tokenize Text ===\n")
text <- "Hello, this is a test"
text_tokens_vec <- chatterbox:::tokenize_text(model$tokenizer, text)
text_tokens <- torch::torch_tensor(text_tokens_vec, dtype = torch::torch_long())$unsqueeze(1L)
cat(sprintf("  Text: '%s'\n", text))
cat(sprintf("  Tokens: %d\n", length(text_tokens_vec)))
cat(sprintf("  Token IDs: %s\n", paste(text_tokens_vec, collapse = ", ")))

# Create T3 conditioning
cat("\n=== Step 3: T3 Conditioning ===\n")
cond <- chatterbox:::t3_cond(
    speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
    emotion_adv = 0.3
)
cat(sprintf("  cond$speaker: %s\n", paste(dim(cond$speaker), collapse = "x")))
cat(sprintf("  cond$cond_prompt_speech_tokens: %s\n", paste(dim(cond$cond_prompt_speech_tokens), collapse = "x")))

# T3 inference
cat("\n=== Step 4: T3 Inference ===\n")
torch::with_no_grad({
        speech_tokens_raw <- chatterbox:::t3_inference(
            model = model$t3,
            cond = cond,
            text_tokens = text_tokens,
            cfg_weight = 0.5,
            temperature = 0.8,
            top_p = 0.8
        )
    })
# Convert to R vector for analysis
speech_tokens_vec <- as.integer(speech_tokens_raw$cpu())
cat(sprintf("  Raw speech tokens: %d tokens\n", length(speech_tokens_vec)))
cat(sprintf("  Token range: [%d, %d]\n", min(speech_tokens_vec), max(speech_tokens_vec)))
cat(sprintf("  First 20: %s\n", paste(head(speech_tokens_vec, 20), collapse = ", ")))

# Check for stop token
stop_idx <- which(speech_tokens_vec == 6562)
if (length(stop_idx) > 0) {
    cat(sprintf("  Stop token (6562) found at position %d\n", stop_idx[1]))
} else {
    cat("  No stop token found\n")
}

# Drop invalid tokens
speech_tokens_clean <- chatterbox:::drop_invalid_tokens(speech_tokens_vec)
cat(sprintf("  After dropping invalid: %d tokens\n", length(speech_tokens_clean)))

if (length(speech_tokens_clean) == 0) {
    cat("ERROR: No valid speech tokens after cleanup\n")
    quit(status = 1)
}

# Convert to tensor
speech_tokens <- torch::torch_tensor(
    as.integer(speech_tokens_clean),
    dtype = torch::torch_long()
)$unsqueeze(1L)
cat(sprintf("  Final speech tokens tensor: %s\n", paste(dim(speech_tokens), collapse = "x")))

# S3Gen inference
cat("\n=== Step 5: S3Gen Inference ===\n")
torch::with_no_grad({
        result <- model$s3gen$inference(
            speech_tokens = speech_tokens,
            ref_dict = voice$ref_dict,
            finalize = TRUE
        )
        audio <- result[[1]]
    })

cat(sprintf("  Output audio shape: %s\n", paste(dim(audio), collapse = "x")))
cat(sprintf("  Output audio mean: %.6f, std: %.6f\n",
        audio$mean()$item(), audio$std()$item()))
cat(sprintf("  Output audio range: [%.4f, %.4f]\n",
        audio$min()$item(), audio$max()$item()))

duration <- audio$size(2) / 24000
cat(sprintf("  Duration: %.2f seconds\n", duration))

# Check for silence
audio_abs_mean <- audio$abs()$mean()$item()
if (audio_abs_mean < 0.01) {
    cat("\nWARNING: Audio is essentially silent (abs mean < 0.01)\n")

    # Debug: check mel output from flow
    cat("\n=== Debug: Check Mel Output ===\n")
    torch::with_no_grad({
            flow_result <- model$s3gen$flow$forward(
                token = speech_tokens,
                token_len = torch::torch_tensor(speech_tokens$size(2)),
                prompt_token = voice$ref_dict$prompt_token,
                prompt_token_len = voice$ref_dict$prompt_token_len,
                prompt_feat = voice$ref_dict$prompt_feat,
                prompt_feat_len = voice$ref_dict$prompt_feat_len,
                embedding = voice$ref_dict$embedding,
                finalize = TRUE
            )
            mel <- flow_result[[1]]
        })

    cat(sprintf("  Mel shape: %s\n", paste(dim(mel), collapse = "x")))
    cat(sprintf("  Mel mean: %.6f, std: %.6f\n", mel$mean()$item(), mel$std()$item()))
    cat(sprintf("  Mel range: [%.4f, %.4f]\n", mel$min()$item(), mel$max()$item()))

    mel_abs_mean <- mel$abs()$mean()$item()
    if (mel_abs_mean < 0.1) {
        cat("\n  DIAGNOSIS: Mel output is near-zero. Issue is in CFM decoder.\n")
    } else {
        cat("\n  DIAGNOSIS: Mel looks OK. Issue might be in vocoder.\n")
    }
} else {
    cat("\nAudio has signal - not silent.\n")
}

cat("\n=== Done ===\n")

