#!/usr/bin/env Rscript
# Trace R conditioning structure for comparison with Python

library(chatterbox)

cat("Loading model...\n")
model <- chatterbox("cpu")
model <- load_chatterbox(model)

text <- "cornball AI is doing something for our country!"
ref_path <- "inst/audio/jfk.wav"

cat(sprintf("\nText: %s\n", text))

# Step 1: Create voice embedding
cat("\n=== Voice Embedding ===\n")
voice <- chatterbox:::create_voice_embedding(model, ref_path)
cat(sprintf("ve_embedding: %s\n", paste(dim(voice$ve_embedding), collapse = "x")))
cat(sprintf("cond_prompt_speech_tokens: %s\n", paste(dim(voice$cond_prompt_speech_tokens), collapse = "x")))

# Step 2: Tokenize text
cat("\n=== Text Tokens ===\n")
text_tokens <- chatterbox:::tokenize_text(model$tokenizer, text)
cat(sprintf("text_tokens: %d tokens\n", length(text_tokens)))
cat(sprintf("token_ids: %s\n", paste(text_tokens, collapse = ", ")))

# Step 3: Create T3 conditioning
cat("\n=== T3 Conditioning ===\n")
cond <- chatterbox:::t3_cond(
    speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
    emotion_adv = 0.5
)
cat(sprintf("speaker_emb: %s\n", paste(dim(cond$speaker_emb), collapse = "x")))
cat(sprintf("cond_prompt_speech_tokens: %s\n", paste(dim(cond$cond_prompt_speech_tokens), collapse = "x")))
cat(sprintf("emotion_adv: %s\n", cond$emotion_adv))

# Step 4: Prepare conditioning (goes through T3 cond encoder)
cat("\n=== Prepared Conditioning ===\n")
prepared_cond <- model$t3$prepare_conditioning(cond)
cat(sprintf("prepared_cond shape: %s\n", paste(dim(prepared_cond), collapse = "x")))
cat(sprintf("prepared_cond mean: %.6f, std: %.6f\n",
        prepared_cond$mean()$item(), prepared_cond$std()$item()))

# Step 5: Prepare full input embeddings
cat("\n=== Full Input Embeddings ===\n")
text_tensor <- torch::torch_tensor(text_tokens, dtype = torch::torch_long())$unsqueeze(1)
# Add start/stop tokens
sot <- model$t3$config$start_text_token# 255
eot <- model$t3$config$stop_text_token# 0
text_tensor <- torch::nnf_pad(text_tensor, c(1, 0), value = sot)
text_tensor <- torch::nnf_pad(text_tensor, c(0, 1), value = eot)
cat(sprintf("text_tensor (with SOT/EOT): %s\n", paste(dim(text_tensor), collapse = "x")))

# BOS speech token
bos_token <- torch::torch_tensor(matrix(model$t3$config$start_speech_token, nrow = 1),
    dtype = torch::torch_long())
cat(sprintf("bos_token: %d\n", as.integer(bos_token$item())))

# Double for CFG
text_tensor_cfg <- torch::torch_cat(list(text_tensor, text_tensor), dim = 1)
bos_token_cfg <- torch::torch_cat(list(bos_token, bos_token), dim = 1)

# Prepare input embeds
prep <- model$t3$prepare_input_embeds(cond, text_tensor_cfg, bos_token_cfg, cfg_weight = 0.5)
cat(sprintf("input_embeds shape: %s\n", paste(dim(prep$embeds), collapse = "x")))
cat(sprintf("len_cond: %d\n", prep$len_cond))
cat(sprintf("input_embeds mean: %.6f, std: %.6f\n",
        prep$embeds$mean()$item(), prep$embeds$std()$item()))

# Breakdown of sequence:
# [conditioning (34)] + [text (16)] + [speech BOS (1)] = 51 positions
# But CFG doubles batch, so shape is [2, 51, 1024]
# Actually: conditioning = 1 speaker + 32 perceiver + 1 emotion = 34
# text = 14 original + 2 (SOT, EOT) = 16
# speech = 1 BOS
# Total = 34 + 16 + 1 = 51
cat(sprintf("\nExpected breakdown:\n"))
cat(sprintf("  Conditioning: 34 (1 speaker + 32 perceiver + 1 emotion)\n"))
cat(sprintf("  Text: %d (original) + 2 (SOT/EOT) = %d\n", length(text_tokens), length(text_tokens) + 2))
cat(sprintf("  Speech BOS: 1\n"))
cat(sprintf("  Total: 34 + %d + 1 = %d\n", length(text_tokens) + 2, 34 + length(text_tokens) + 2 + 1))

