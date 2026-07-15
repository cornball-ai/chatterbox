#!/usr/bin/env r
# bonsaisitter structural audit: numeric constants that drifted between a
# torch reference scope and its anvl/yunque port. Extend `pairings` as
# components are ported.
suppressMessages(library(bonsaisitter))

lines_of <- function(path, from, to) {
  paste(readLines(path)[from:to], collapse = "\n")
}

audit <- function(label, reference, port) {
  a <- audit_translation(reference, port, lang = "r")
  cat(sprintf("\n== %s ==\n", label))
  cat("literals in reference not in port:",
    if (length(a$literals_missing)) paste(a$literals_missing, collapse = ", ") else "(none)",
    "\n")
  cat("literals in port not in reference:",
    if (length(a$literals_extra)) paste(a$literals_extra, collapse = ", ") else "(none)",
    "\n")
}

# --- pairings ---
audit("voice encoder",
  lines_of("R/voice_encoder.R", 130, 152),
  paste(readLines("R/yq_voice_encoder.R"), collapse = "\n"))

# T3 Llama backbone: config + RoPE + attention/mlp/layer forwards
audit("llama backbone",
  paste(lines_of("R/llama.R", 18, 40), lines_of("R/llama.R", 82, 143),
    lines_of("R/llama.R", 211, 361), sep = "\n"),
  paste(readLines("R/yq_llama.R"), collapse = "\n"))

# T3 conditioning encoder: attention_block + perceiver + cond_enc forwards
audit("t3 cond enc",
  paste(lines_of("R/t3.R", 160, 197), lines_of("R/t3.R", 230, 244),
    lines_of("R/t3.R", 282, 319), sep = "\n"),
  paste(readLines("R/yq_t3_cond.R"), collapse = "\n"))

# T3 forward: prepare_input_embeds + forward + heads
audit("t3 forward",
  lines_of("R/t3.R", 366, 447),
  paste(readLines("R/yq_t3.R"), collapse = "\n"))

# CAMPPlus speaker xvector: blocks + pooling + model forward
audit("campplus",
  lines_of("R/speaker_encoder.R", 12, 428),
  paste(readLines("R/yq_campplus.R"), collapse = "\n"))

# S3 tokenizer: rope + FSQ + FSMN attention + encoder forward
audit("s3tokenizer",
  lines_of("R/s3tokenizer.R", 9, 657),
  paste(readLines("R/yq_s3tokenizer.R"), collapse = "\n"))

# Upsample conformer encoder (flow.encoder)
audit("conformer",
  lines_of("R/conformer.R", 18, 612),
  paste(readLines("R/yq_conformer.R"), collapse = "\n"))

# Mel + Kaldi fbank frontends (incl. VE variant + S3 tokenizer log-mel)
audit("mel_fbank",
  paste(lines_of("R/audio_utils.R", 194, 370),
    lines_of("R/kaldi_fbank.R", 7, 210),
    lines_of("R/s3tokenizer.R", 138, 165), sep = "\n"),
  paste(readLines("R/yq_mel_fbank.R"), collapse = "\n"))

# Windowed-sinc resampler
audit("resample",
  lines_of("R/resample.R", 36, 165),
  paste(readLines("R/yq_resample.R"), collapse = "\n"))
