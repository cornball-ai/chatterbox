#!/usr/bin/env r
# VRAM profiling: measure GPU memory at each stage of inference
#
# Uses libtorch's CUDACachingAllocator stats for accurate per-process
# measurement (allocated = active tensors, reserved = allocated + cache).

library(chatterbox)

MB <- 1024^2

report <- function (label) {
    gc(); gc()
    stats <- Rtorch::cuda_memory_stats()
    alloc <- stats["allocated_current"] / MB
    resrv <- stats["reserved_current"] / MB
    cat(sprintf("  %-45s %6.0f MB alloc  %6.0f MB reserved\n", label, alloc, resrv))
    c(alloc = alloc, reserved = resrv)
}

cat("============================================================\n")
cat("             VRAM Profile: Chatterbox on CUDA\n")
cat("============================================================\n")
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))
cat(sprintf("Total VRAM: %.0f MB\n\n",
    Rtorch::cuda_mem_info()["total"] / MB))

text <- "The quick brown fox jumps over the lazy dog."
ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")

cat("Stage-by-stage GPU memory (allocated / reserved):\n")
mem0 <- report("Baseline")

model <- chatterbox("cuda")
paths <- get_model_paths()

# Tokenizer (CPU only)
model$tokenizer <- chatterbox:::load_bpe_tokenizer(paths$tokenizer)
report("Tokenizer loaded (CPU)")

# Voice encoder
ve_weights <- chatterbox:::read_safetensors(paths$ve, "cpu")
model$voice_encoder <- chatterbox:::voice_encoder()
model$voice_encoder <- chatterbox:::load_voice_encoder_weights(model$voice_encoder, ve_weights)
rm(ve_weights); gc()
report("Voice encoder loaded (CPU)")
model$voice_encoder$to(device = "cuda")
model$voice_encoder$eval()
mem_ve <- report("Voice encoder on CUDA")

# T3 model
t3_weights <- chatterbox:::read_safetensors(paths$t3_cfg, "cpu")
model$t3 <- chatterbox:::t3_model()
model$t3 <- chatterbox:::load_t3_weights(model$t3, t3_weights)
rm(t3_weights); gc()
report("T3 loaded (CPU)")
model$t3$to(device = "cuda")
model$t3$eval()
mem_t3 <- report("T3 on CUDA")

# S3Gen
model$s3gen <- chatterbox:::load_s3gen(paths$s3gen, "cuda")
model$loaded <- TRUE
mem_s3 <- report("S3Gen on CUDA (all loaded)")

# Voice embedding
voice <- create_voice_embedding(model, ref_audio)
gc()
mem_voice <- report("After voice embedding")

# T3 inference
text_tokens <- chatterbox:::tokenize_text(model$tokenizer, text)
text_tokens <- Rtorch::torch_tensor(text_tokens, dtype = Rtorch::torch_long)$unsqueeze(1)$to(device = "cuda")
cond <- chatterbox:::t3_cond(
    speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
    emotion_adv = 0.5
)
Rtorch::with_no_grad({
    speech_tokens <- chatterbox:::t3_inference(
        model = model$t3, cond = cond,
        text_tokens = text_tokens,
        cfg_weight = 0.5, temperature = 0.8, top_p = 0.9
    )
})
gc()
mem_t3_inf <- report("After T3 inference")

# Offload T3 + VE
model$t3$to(device = "cpu")
model$voice_encoder$to(device = "cpu")
rm(cond, text_tokens); gc(); gc()
report("After T3/VE offloaded (cache dirty)")
Rtorch::cuda_empty_cache()
mem_offload <- report("After T3/VE offloaded + cache clear")

# S3Gen inference
speech_tokens <- chatterbox:::drop_invalid_tokens(speech_tokens)
speech_tokens_t <- Rtorch::torch_tensor(
    as.integer(speech_tokens), dtype = Rtorch::torch_long
)$unsqueeze(1)$to(device = "cuda")

Rtorch::with_no_grad({
    result <- model$s3gen$inference(
        speech_tokens = speech_tokens_t,
        ref_dict = voice$ref_dict,
        finalize = TRUE
    )
})
gc()
mem_s3_inf <- report("After S3Gen inference")

audio <- as.numeric(result[[1]]$squeeze()$cpu())
rm(result, speech_tokens_t); gc()
Rtorch::cuda_empty_cache()
mem_post <- report("After audio extracted + cache clear")

# Restore T3/VE
model$t3$to(device = "cuda")
model$voice_encoder$to(device = "cuda")
gc()
mem_restore <- report("After T3/VE restored to CUDA")

# Summary
cat("\n")
cat("============================================================\n")
cat("                       SUMMARY\n")
cat("============================================================\n")
cat(sprintf("Voice encoder weights:       %5.0f MB\n", mem_ve[1]))
cat(sprintf("T3 weights:                  %5.0f MB\n", mem_t3[1] - mem_ve[1]))
cat(sprintf("S3Gen weights:               %5.0f MB\n", mem_s3[1] - mem_t3[1]))
cat(sprintf("All weights (allocated):     %5.0f MB\n", mem_s3[1]))
cat(sprintf("Voice embedding overhead:    %5.0f MB\n", mem_voice[1] - mem_s3[1]))
cat(sprintf("T3 inference peak reserved:  %5.0f MB\n", mem_t3_inf[2]))
cat(sprintf("After T3/VE offload+clear:   %5.0f MB alloc, %5.0f MB reserved\n",
    mem_offload[1], mem_offload[2]))
cat(sprintf("S3Gen inference peak resrvd: %5.0f MB\n", mem_s3_inf[2]))
cat(sprintf("Post-inference + clear:      %5.0f MB alloc, %5.0f MB reserved\n",
    mem_post[1], mem_post[2]))
cat(sprintf("Steady state (all on GPU):   %5.0f MB alloc, %5.0f MB reserved\n",
    mem_restore[1], mem_restore[2]))
cat(sprintf("\nGenerated %.1fs audio (%d tokens)\n",
    length(audio) / 24000, length(speech_tokens)))
