#!/usr/bin/env r
# Profile T3 inference to find optimization opportunities

library(torch)

rhydrogen::load_all("/home/troy/chatterbox")

# Load model
model <- chatterbox("cuda")
model <- load_chatterbox(model)

# Prepare inputs
ref_audio <- "/home/troy/Music/cornball_jfk.wav"
voice <- create_voice_embedding(model, ref_audio)

text <- "Hello, this is a test."
text_tokens <- tokenize_text(model$tokenizer, text)
text_tokens <- torch::torch_tensor(text_tokens, dtype = torch::torch_long())$unsqueeze(1)$to(device = "cuda")

cond <- t3_cond(
    speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
    emotion_adv = 0.5
)

# Warm up
cat("Warming up...\n")
invisible(t3_inference(model$t3, cond, text_tokens, max_new_tokens = 10))

# Profile with timing
cat("\n=== Profiling T3 Inference ===\n")

# Time the full inference
torch::cuda_synchronize()
start <- Sys.time()
tokens <- t3_inference(model$t3, cond, text_tokens, max_new_tokens = 100)
torch::cuda_synchronize()
total_time <- as.numeric(Sys.time() - start)

cat(sprintf("Generated %d tokens in %.2f seconds\n", length(tokens), total_time))
cat(sprintf("Average: %.1f ms/token\n", total_time * 1000 / length(tokens)))

# Now let's profile individual components
cat("\n=== Component Timing (single forward pass) ===\n")

# Get model internals
t3 <- model$t3
config <- t3$config
device <- "cuda"

# Prepare a single step
bos_token <- torch::torch_tensor(matrix(config$start_speech_token, nrow = 1),
    device = device, dtype = torch::torch_long())
bos_token <- torch::torch_cat(list(bos_token, bos_token), dim = 1)  # CFG doubling

# Time embedding lookup
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:10) {
    emb <- t3$speech_emb$forward(bos_token$add(1L))
    pos <- t3$speech_pos_emb$get_fixed_embedding(0)
    combined <- emb + pos
}
torch::cuda_synchronize()
emb_time <- as.numeric(Sys.time() - t1) / 10 * 1000
cat(sprintf("Embedding lookup: %.2f ms\n", emb_time))

# Time transformer forward (the big one)
# First, build full input embeds
text_tokens_cfg <- torch::torch_cat(list(text_tokens, text_tokens), dim = 1)
cond_cfg <- t3_cond(
    speaker_emb = torch::torch_cat(list(cond$speaker_emb, cond$speaker_emb), dim = 1),
    cond_prompt_speech_tokens = torch::torch_cat(list(cond$cond_prompt_speech_tokens, cond$cond_prompt_speech_tokens), dim = 1),
    emotion_adv = 0.5
)
cond_emb <- t3$cond_enc$forward(cond_cfg)

text_emb <- t3$text_emb$forward(text_tokens_cfg$add(1L))
text_emb <- text_emb + t3$text_pos_emb$forward(text_tokens_cfg)
speech_emb <- t3$speech_emb$forward(bos_token$add(1L)) + t3$speech_pos_emb$get_fixed_embedding(0)

embeds <- torch::torch_cat(list(cond_emb, text_emb, speech_emb), dim = 2)

torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:10) {
    output <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
}
torch::cuda_synchronize()
tfmr_time <- as.numeric(Sys.time() - t1) / 10 * 1000
cat(sprintf("Transformer forward (initial, with cache): %.2f ms\n", tfmr_time))

# Time transformer forward with KV cache (subsequent steps)
past_kv <- output$past_key_values
next_emb <- t3$speech_emb$forward(bos_token$add(1L)) + t3$speech_pos_emb$get_fixed_embedding(1)

torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:10) {
    output2 <- t3$tfmr$forward(inputs_embeds = next_emb, past_key_values = past_kv, use_cache = TRUE)
}
torch::cuda_synchronize()
tfmr_cached_time <- as.numeric(Sys.time() - t1) / 10 * 1000
cat(sprintf("Transformer forward (cached, single token): %.2f ms\n", tfmr_cached_time))

# Time head projection
hidden <- output2$last_hidden_state
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:10) {
    logits <- t3$speech_head$forward(hidden[, , -1, drop = FALSE])
}
torch::cuda_synchronize()
head_time <- as.numeric(Sys.time() - t1) / 10 * 1000
cat(sprintf("Speech head projection: %.2f ms\n", head_time))

# Time sampling (top-p)
logits_2d <- logits$squeeze(2)
torch::cuda_synchronize()
t1 <- Sys.time()
for (i in 1:10) {
    probs <- torch::nnf_softmax(logits_2d / 0.8, dim = -1)
    sorted <- torch::torch_sort(probs, descending = TRUE)
    sorted_probs <- sorted[[1]]
    sorted_indices <- sorted[[2]]
    cumsum <- torch::torch_cumsum(sorted_probs, dim = -1)
    mask <- cumsum > 0.95
    # Would do sampling here
}
torch::cuda_synchronize()
sample_time <- as.numeric(Sys.time() - t1) / 10 * 1000
cat(sprintf("Sampling (softmax + sort + cumsum): %.2f ms\n", sample_time))

cat("\n=== Summary ===\n")
cat(sprintf("Per-token breakdown estimate:\n"))
cat(sprintf("  Embedding:    %.2f ms\n", emb_time))
cat(sprintf("  Transformer:  %.2f ms (cached)\n", tfmr_cached_time))
cat(sprintf("  Head:         %.2f ms\n", head_time))
cat(sprintf("  Sampling:     %.2f ms\n", sample_time))
cat(sprintf("  ---------------\n"))
cat(sprintf("  Estimated:    %.2f ms\n", emb_time + tfmr_cached_time + head_time + sample_time))
cat(sprintf("  Actual:       %.1f ms\n", total_time * 1000 / length(tokens)))
