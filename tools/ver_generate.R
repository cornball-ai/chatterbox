#!/usr/bin/env r
# Verify the anvl e2e generation chain against the torch fixture.
# anvl/yunque only (no torch in this process).
suppressMessages({
  library(anvl)
  library(yunque)
})

e <- new.env()
for (f in c("yq_common.R", "yq_resample.R", "yq_mel_fbank.R",
  "yq_voice_encoder.R", "yq_s3tokenizer.R", "yq_campplus.R", "yq_llama.R",
  "yq_t3_cond.R", "yq_t3.R", "yq_conformer.R", "yq_cfm.R", "yq_flow.R",
  "yq_hifigan.R", "yq_tts.R")) {
  sys.source(file.path("R", f), envir = e)
}

fx <- readRDS("tools/fixtures/generate.rds")
w <- list(
  ve = e$yq_ve_load_weights(fx$ve_path),
  campplus = e$yq_campplus_load_weights(fx$s3gen_path),
  s3tok = e$yq_s3tokenizer_load_weights(fx$s3gen_path),
  t3 = e$yq_t3_load_weights(fx$t3_path),
  cond = e$yq_t3_cond_load_weights(fx$t3_path),
  llama = e$yq_llama_load_weights(fx$t3_path),
  flow = e$yq_flow_load_weights(fx$s3gen_path),
  hifigan = e$yq_hifigan_load_weights(fx$s3gen_path)
)

reltol <- function(a, b) max(abs(a - b)) / max(abs(b))
ok <- TRUE

# Voice embedding (re-verified here so the chain is torch-free end to end).
voice <- e$yq_voice_embedding(fx$samples, fx$sr, w)

# Stage A (gating): torch speech tokens + captured RNG -> mel + waveform.
res <- e$yq_generate(NULL, voice, w, flow_noise = fx$flow_noise,
  source_phase = fx$phase, source_noise = fx$z1,
  speech_tokens = fx$tokens_clean)

mel_got <- as.array(res$mel)
r_mel <- reltol(mel_got, fx$mel)
p_mel <- identical(dim(mel_got), dim(fx$mel)) && r_mel < 1e-3
ok <- ok && p_mel
cat(sprintf("[mel   ] %s  reltol %.3e  dim %s\n",
  if (p_mel) "PASS" else "FAIL", r_mel, paste(dim(mel_got), collapse = "x")))

audio_got <- matrix(res$audio, nrow = 1L)
r_aud <- reltol(audio_got, fx$audio)
p_aud <- identical(dim(audio_got), dim(fx$audio)) && r_aud < 1e-2
ok <- ok && p_aud
cat(sprintf("[audio ] %s  reltol %.3e  n %d\n",
  if (p_aud) "PASS" else "FAIL", r_aud, length(audio_got)))

# Stage B (smoke, non-gating): run the T3 sampling loop for 16 tokens.
# Token equality with torch is NOT expected -- the reference samples with
# torch_multinomial (torch RNG), this port with sample.int (R RNG); the
# sampler math itself is mirrored step for step. Coincidences happen where
# the distribution is concentrated.
tt <- matrix(c(255L, fx$text_ids, 0L), nrow = 1L)
prompt_emb <- e$.yq_t3_embed(w$t3$speech_emb, w$t3$speech_pos,
  matrix(voice$cond_prompt_speech_tokens, nrow = 1L))
cond_emb <- e$yq_t3_cond_enc(
  anvl::nv_array(voice$ve_embedding, dtype = "f32"), prompt_emb, 0.5, w$cond)
set.seed(778)
tok16 <- e$yq_t3_generate(cond_emb, tt, w$t3, w$llama, e$.yq_t3_config(),
  max_new = 16L)
n <- min(length(tok16), length(fx$tokens16))
cat(sprintf(
  "[t3 loop] smoke: %d tokens sampled (%d/%d coincide with torch; RNG streams differ by design)\n",
  length(tok16),
  if (n > 0) sum(tok16[seq_len(n)] == fx$tokens16[seq_len(n)]) else 0L,
  length(fx$tokens16)))

quit(status = if (ok) 0L else 1L)
