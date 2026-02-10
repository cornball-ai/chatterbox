#!/usr/bin/env r
# Detailed VRAM profiling: instrument each step of generate()

library(torch)
library(chatterbox)

vram <- function(label) {
    smi <- system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", intern = TRUE)
    cat(sprintf("%-55s  %5s MB\n", label, trimws(smi[1])))
}

device <- "cuda"
ref_audio <- system.file("audio", "jfk.mp3", package = "chatterbox")
text <- "The quick brown fox jumps over the lazy dog."

cat("=== Detailed VRAM Profile ===\n\n")
vram("Baseline")

model <- chatterbox(device)
model <- load_chatterbox(model)
gc()
vram("Model loaded + gc()")

voice <- create_voice_embedding(model, ref_audio)
gc()
vram("Voice embedding + gc()")

# ============================================================
# Manually step through generate() path
# ============================================================

cat("\n--- T3 inference (step by step) ---\n")

text_tokens <- chatterbox:::tokenize_text(model$tokenizer, text)
text_tokens <- torch_tensor(text_tokens, dtype = torch_long())$unsqueeze(1L)$to(device = device)

cond <- chatterbox:::t3_cond(
    speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
    emotion_adv = 0.5
)

t3 <- model$t3
config <- t3$config
cfg_weight <- 0.5

# Prepare inputs (same as t3_inference)
sot <- config$start_text_token
eot <- config$stop_text_token
text_tokens <- nnf_pad(text_tokens, c(1L, 0L), value = sot)
text_tokens <- nnf_pad(text_tokens, c(0L, 1L), value = eot)
text_tokens <- torch_cat(list(text_tokens, text_tokens), dim = 1L)

bos_token <- torch_tensor(matrix(config$start_speech_token, nrow = 1L),
    device = device, dtype = torch_long())
bos_token <- torch_cat(list(bos_token, bos_token), dim = 1L)

prep <- t3$prepare_input_embeds(cond, text_tokens, bos_token, cfg_weight)
embeds <- prep$embeds
gc()
vram("After prepare_input_embeds + gc()")

# Prefill
with_no_grad({
    output <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
    past_key_values <- output$past_key_values
})
gc()
vram("After prefill (tfmr forward + KV cache)")

# Check KV cache size
n_layers <- length(past_key_values)
kv_shape <- dim(past_key_values[[1]][[1]])
kv_bytes <- n_layers * 2 * prod(kv_shape) * 4  # float32
cat(sprintf("  KV cache: %d layers x 2 x %s = %.0f MB\n",
    n_layers, paste(kv_shape, collapse="x"), kv_bytes / 1024^2))

# Generation loop (just a few tokens to see the pattern)
cat("\n--- Generation loop (10 tokens) ---\n")
generated_ids <- bos_token[1L,, drop = FALSE]$clone()

with_no_grad({
    for (i in 1:10) {
        logits <- output$last_hidden_state[, -1L, ]
        logits <- t3$speech_head$forward(logits)
        cond_logits <- logits[1L, ]$unsqueeze(1L)
        uncond_logits <- logits[2L, ]$unsqueeze(1L)
        logits <- cond_logits + 0.5 * (cond_logits - uncond_logits)

        probs <- nnf_softmax(logits, dim = -1L)
        next_token <- torch_multinomial(probs$squeeze(1L), num_samples = 1L)$unsqueeze(1L)
        generated_ids <- torch_cat(list(generated_ids, next_token), dim = 2L)

        next_emb <- t3$speech_emb$forward(next_token) +
            t3$speech_pos_emb$get_fixed_embedding(i)
        next_emb <- torch_cat(list(next_emb, next_emb), dim = 1L)

        output <- t3$tfmr$forward(inputs_embeds = next_emb,
            past_key_values = past_key_values, use_cache = TRUE)
        past_key_values <- output$past_key_values
    }
})
gc()
vram("After 10 token generation loop + gc()")

# Check KV cache size now (should have grown)
kv_shape2 <- dim(past_key_values[[1]][[1]])
kv_bytes2 <- n_layers * 2 * prod(kv_shape2) * 4
cat(sprintf("  KV cache now: %d layers x 2 x %s = %.0f MB\n",
    n_layers, paste(kv_shape2, collapse="x"), kv_bytes2 / 1024^2))

# Clear the generation intermediates
cat("\n--- Cleanup intermediates ---\n")
rm(output, past_key_values, embeds, prep, logits, probs, next_token,
   next_emb, generated_ids, cond_logits, uncond_logits,
   text_tokens, bos_token, cond)
gc()
vram("After rm intermediates + gc()")

# Now run S3Gen
cat("\n--- S3Gen inference ---\n")
speech_tokens <- torch_tensor(rep(100L, 50), dtype = torch_long())$unsqueeze(1L)$to(device = device)
vram("Before S3Gen")

with_no_grad({
    result <- model$s3gen$inference(
        speech_tokens = speech_tokens,
        ref_dict = voice$ref_dict,
        finalize = TRUE,
        traced = FALSE
    )
})
gc()
vram("After S3Gen + gc()")

rm(result, speech_tokens)
gc()
vram("After rm S3Gen result + gc()")

# Final cleanup
cat("\n--- Full cleanup ---\n")
rm(t3, model, voice)
gc()
vram("After rm(model, voice, t3) + gc()")
