#!/usr/bin/env Rscript --vanilla
# E2E generation fixture: the full torch chain (text -> T3 speech tokens ->
# CFM mel -> HiFT waveform) conditioned on jfk.wav. Torch only (CPU).
library(chatterbox)

model <- chatterbox(device = "cpu")
aud <- chatterbox:::read_audio("inst/audio/jfk.wav")
voice <- torch::with_no_grad({
  create_voice_embedding(model, "inst/audio/jfk.wav")
})

text <- "Ask not what your country can do for you."
norm <- chatterbox:::normalize_tts_text(text)
text_ids <- chatterbox:::tokenize_text(model$tokenizer, norm)

tt <- torch::torch_tensor(text_ids, dtype = torch::torch_long())$unsqueeze(1L)
cond <- chatterbox:::t3_cond(
  speaker_emb = voice$ve_embedding,
  cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
  emotion_adv = 0.5)

# Full-quality T3 run (R RNG drives the sampling).
set.seed(777)
tok <- torch::with_no_grad({
  chatterbox:::t3_inference(model$t3, cond, tt, max_new_tokens = 1000)
})
tok_clean <- as.integer(chatterbox:::drop_invalid_tokens(tok)$cpu())
cat(sprintf("T3 generated %d tokens (%d after drop_invalid)\n",
  tok$size(1), length(tok_clean)))

# Capped run for the anvl sampling-loop comparison (informational).
set.seed(778)
tok16 <- torch::with_no_grad({
  chatterbox:::t3_inference(model$t3, cond, tt, max_new_tokens = 16L)
})
tok16 <- as.integer(tok16$cpu())

# Flow noise: the slice of the flow's rand_noise buffer the CFM consumes.
n_prompt <- as.integer(voice$ref_dict$prompt_token$size(2))
mel_total <- 2L * (n_prompt + length(tok_clean))
flow_noise <- as.array(model$s3gen$flow$decoder$rand_noise$cpu())[1, , seq_len(mel_total)]

# Mel (skip_vocoder; the standard flow path consumes no torch RNG).
st <- torch::torch_tensor(matrix(tok_clean, nrow = 1L),
  dtype = torch::torch_long())
mel_out <- torch::with_no_grad({
  model$s3gen$inference(speech_tokens = st, ref_dict = voice$ref_dict,
    finalize = TRUE, skip_vocoder = TRUE)
})
mel <- as.array(mel_out[[1]]$cpu())

# Waveform with seeded vocoder RNG, then replay the draws it consumed:
# sine_gen's uniform phase, its randn source noise, the discarded randn.
torch::torch_manual_seed(4321)
wav_out <- torch::with_no_grad({
  model$s3gen$inference(speech_tokens = st, ref_dict = voice$ref_dict,
    finalize = TRUE)
})
audio <- as.array(wav_out[[1]]$cpu())

T_wav <- dim(mel)[3] * 480L
torch::torch_manual_seed(4321)
phase <- as.array(torch::torch_empty(c(1L, 9L, 1L))$uniform_(-pi, pi))
z1 <- as.array(torch::torch_randn(c(1L, 9L, T_wav)))

paths <- chatterbox:::get_model_paths()
dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(
  samples = aud$samples, sr = aud$sr, text = text, text_ids = text_ids,
  tokens_clean = tok_clean, tokens16 = tok16,
  flow_noise = flow_noise, mel = mel, audio = audio,
  phase = phase, z1 = z1,
  ve_path = paths$ve, s3gen_path = paths$s3gen, t3_path = paths$t3_cfg),
  "tools/fixtures/generate.rds")
cat(sprintf("generate fixture: %d tokens, mel %s, audio %s\n",
  length(tok_clean), paste(dim(mel), collapse = "x"),
  paste(dim(audio), collapse = "x")))
