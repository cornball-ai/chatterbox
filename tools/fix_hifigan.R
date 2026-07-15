#!/usr/bin/env Rscript --vanilla
# Fixture generator for the HiFT/HiFiGAN vocoder. Torch only (CPU).
library(chatterbox)

paths <- chatterbox:::get_model_paths()

model <- chatterbox:::create_s3gen_vocoder("cpu")
sd <- safetensors::safe_load_file(paths$s3gen, framework = "torch",
  device = "cpu")
torch::with_no_grad({
  model <- chatterbox:::load_hifigan_weights(model, sd, prefix = "mel2wav.")
})
model$eval()

set.seed(1234)
B <- 1L
H <- 9L # nb_harmonics + 1

# Real speech mel: the trained f0 predictor collapses to ~0 Hz on random
# mels, so a real mel keeps the input scale plausible end-to-end.
a <- chatterbox:::read_audio(system.file("audio", "jfk.wav",
  package = "chatterbox"))
w24 <- chatterbox:::resample_audio(a$samples, a$sr, 24000)
mel_full <- torch::with_no_grad(chatterbox:::compute_mel_spectrogram(w24))
mel_t <- mel_full[, , 101:148]$contiguous() # [1, 80, 48]
mel <- as.array(mel_t)

# Check 1: f0 predictor (deterministic).
f0_t <- torch::with_no_grad(model$f0_predictor$forward(mel_t))
f0 <- as.array(f0_t)
T_wav <- ncol(f0) * 480L
cat("f0 range:", range(f0), "voiced frac:", mean(f0 > 10), "\n")

# Check 2: full inference mel -> waveform, seeded torch RNG.
torch::torch_manual_seed(4321)
out <- torch::with_no_grad(model$inference(mel_t))
audio <- as.array(out$audio)
src <- as.array(out$source)

# Replay the RNG draws inference consumed, in order: sine_gen's uniform
# initial phase, sine_gen's randn source noise, then the source module's
# extra randn that inference discards.
torch::torch_manual_seed(4321)
phase <- torch::torch_empty(c(B, H, 1L))$uniform_(-pi, pi)
z1 <- torch::torch_randn(c(B, H, T_wav))
z2 <- torch::torch_randn(c(B, T_wav, 1L)) # consumed, unused downstream

# Prove the replay is aligned: rerun the actual sine_gen with the same
# seed; its returned noise must equal noise_amp * z1 bitwise.
f0_up <- torch::with_no_grad(
  model$f0_upsamp$forward(f0_t$unsqueeze(2))) # [B, 1, T_wav]
torch::torch_manual_seed(4321)
sg <- torch::with_no_grad(model$m_source$l_sin_gen$forward(f0_up))
namp <- sg$uv * 0.003 + (1 - sg$uv) * 0.1 / 3
stopifnot(as.numeric(torch::torch_max(
  torch::torch_abs(sg$noise - namp * z1))) == 0)

# Check 3: source module on a synthetic f0 sweep. The trained predictor
# stays < 1 Hz on any mel (all-unvoiced), so the voiced sine path needs
# an injected f0 to be exercised at all.
T3 <- 4800L
f0_syn <- c(seq(0, 400, length.out = 2400L), rep(0, 480L),
  seq(300, 80, length.out = 1920L))
f0_syn_t <- torch::torch_tensor(array(f0_syn, c(1L, T3, 1L)),
  dtype = torch::torch_float())
torch::torch_manual_seed(9999)
src3 <- torch::with_no_grad(model$m_source$forward(f0_syn_t))
sine_merge3 <- as.array(src3$sine_merge) # [1, T3, 1]

torch::torch_manual_seed(9999)
phase3 <- torch::torch_empty(c(B, H, 1L))$uniform_(-pi, pi)
z3 <- torch::torch_randn(c(B, H, T3))

# Alignment proof for the check-3 draws.
torch::torch_manual_seed(9999)
sg3 <- torch::with_no_grad(
  model$m_source$l_sin_gen$forward(f0_syn_t$transpose(2, 3)))
namp3 <- sg3$uv * 0.003 + (1 - sg3$uv) * 0.1 / 3
stopifnot(as.numeric(torch::torch_max(
  torch::torch_abs(sg3$noise - namp3 * z3))) == 0)

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(
  mel = mel, f0 = f0, audio = audio, source = src,
  phase = as.array(phase), z1 = as.array(z1),
  f0_syn = as.array(f0_syn_t$squeeze(3)), # [1, T3], float32-exact
  sine_merge3 = sine_merge3,
  phase3 = as.array(phase3), z3 = as.array(z3),
  s3gen = paths$s3gen),
  "tools/fixtures/hifigan.rds")
cat("saved fixture: f0", paste(dim(f0), collapse = "x"),
  "audio", paste(dim(audio), collapse = "x"),
  "source", paste(dim(src), collapse = "x"),
  "sine_merge3", paste(dim(sine_merge3), collapse = "x"), "\n")
