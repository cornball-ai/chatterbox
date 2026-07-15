#!/usr/bin/env Rscript
# Torch reference fixtures for the anvl/yunque chatterbox port. Run with the
# torch chatterbox installed. CPU-only. Writes tools/fixtures/*.rds. Kept
# separate from the anvl verify so torch and anvl never co-load.
suppressMessages(library(chatterbox))

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
set.seed(1234)
paths <- chatterbox:::get_model_paths()

# --- voice encoder (3-layer LSTM speaker embedding) ---
cfg <- chatterbox:::voice_encoder_config()
ve <- chatterbox:::voice_encoder(cfg)
sd <- safetensors::safe_load_file(paths$ve, framework = "torch", device = "cpu")
chatterbox:::load_voice_encoder_weights(ve, sd)
ve$eval()
n_frames <- 120L
mels <- array(rnorm(1 * n_frames * cfg$num_mels), c(1L, n_frames, cfg$num_mels))
emb <- as.array(torch::with_no_grad(
  ve$forward(torch::torch_tensor(mels, dtype = torch::torch_float()))))
saveRDS(list(mels = mels, emb = emb, ve_path = paths$ve),
  "tools/fixtures/voice_encoder.rds")
cat(sprintf("voice_encoder fixture: emb %s\n", paste(dim(emb), collapse = "x")))

# --- T3 Llama backbone (tfmr.*) ---
lm <- chatterbox:::llama_model(chatterbox:::llama_config_520m())
sd <- safetensors::safe_load_file(paths$t3_cfg, framework = "torch",
  device = "cpu")
tsd <- sd[grepl("^tfmr\\.", names(sd))]
names(tsd) <- sub("^tfmr\\.", "", names(tsd))
lm$load_state_dict(tsd, strict = FALSE)
lm$eval()
seq <- 16L
embeds <- array(rnorm(1 * seq * 1024) * 0.1, c(1L, seq, 1024L))
out <- as.array(torch::with_no_grad(lm$forward(
  inputs_embeds = torch::torch_tensor(embeds, dtype = torch::torch_float()),
  use_cache = FALSE)$last_hidden_state))
saveRDS(list(embeds = embeds, out = out, t3_path = paths$t3_cfg),
  "tools/fixtures/llama.rds")
cat(sprintf("llama fixture: out %s\n", paste(dim(out), collapse = "x")))

# --- T3 conditioning encoder (cond_enc.*) ---
ce <- chatterbox:::t3_cond_enc(chatterbox:::t3_config_english())
csd <- sd[grepl("^cond_enc\\.", names(sd))]
names(csd) <- sub("^cond_enc\\.", "", names(csd))
ce$load_state_dict(csd, strict = FALSE)
ce$eval()
spk <- array(rnorm(1 * 256) * 0.1, c(1L, 256L))
prompt <- array(rnorm(1 * 20 * 1024) * 0.1, c(1L, 20L, 1024L))
emo <- 0.5
cond <- list(
  speaker_emb = torch::torch_tensor(spk, dtype = torch::torch_float()),
  cond_prompt_speech_emb = torch::torch_tensor(prompt,
    dtype = torch::torch_float()),
  emotion_adv = emo)
cout <- as.array(torch::with_no_grad(ce$forward(cond)))
saveRDS(list(spk = spk, prompt = prompt, emo = emo, out = cout,
  t3_path = paths$t3_cfg), "tools/fixtures/t3_cond.rds")
cat(sprintf("t3_cond fixture: out %s\n", paste(dim(cout), collapse = "x")))

# --- T3 forward (speech logits) ---
t3 <- chatterbox:::t3_model()
t3 <- chatterbox:::load_t3_weights(t3, sd)
t3$eval()
tt <- matrix(sample(0:703, 8L, replace = TRUE), nrow = 1L)
stk <- matrix(sample(0:8193, 6L, replace = TRUE), nrow = 1L)
t3out <- as.array(torch::with_no_grad(t3$forward(cond,
  torch::torch_tensor(tt, dtype = torch::torch_long()),
  torch::torch_tensor(stk, dtype = torch::torch_long()))$speech_logits))
saveRDS(list(spk = spk, prompt = prompt, emo = emo, text_tokens = tt,
  speech_tokens = stk, speech_logits = t3out, t3_path = paths$t3_cfg),
  "tools/fixtures/t3.rds")
cat(sprintf("t3 fixture: speech_logits %s\n", paste(dim(t3out), collapse = "x")))
