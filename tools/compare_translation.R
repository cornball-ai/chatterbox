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

# T3 forward + generation: prepare_input_embeds, forward, heads, sampler,
# t3_inference loop
audit("t3 forward",
  lines_of("R/t3.R", 366, 660),
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

# Mel + Kaldi fbank frontends (incl. VE power mel + S3 tokenizer log-mel).
# compute_ve_mel (voice_encoder.R) is the live VE path; audio_utils'
# compute_mel_spectrogram_ve is dead code and deliberately not ported.
audit("mel_fbank",
  paste(lines_of("R/audio_utils.R", 194, 354),
    lines_of("R/voice_encoder.R", 11, 97),
    lines_of("R/kaldi_fbank.R", 7, 210),
    lines_of("R/s3tokenizer.R", 138, 165), sep = "\n"),
  paste(readLines("R/yq_mel_fbank.R"), collapse = "\n"))

# Windowed-sinc resampler
audit("resample",
  lines_of("R/resample.R", 36, 165),
  paste(readLines("R/yq_resample.R"), collapse = "\n"))

# CFM: pos/time embeddings, causal blocks, attention/FF, estimator, solver
# (traced branches 686-717/739-745/772-774 are torch-only and excluded)
audit("cfm",
  paste(lines_of("R/s3gen.R", 60, 532), lines_of("R/s3gen.R", 552, 579),
    lines_of("R/s3gen.R", 606, 685), lines_of("R/s3gen.R", 718, 738),
    lines_of("R/s3gen.R", 746, 771), lines_of("R/s3gen.R", 775, 782),
    sep = "\n"),
  paste(readLines("R/yq_cfm.R"), collapse = "\n"))

# Flow wrapper: make_pad_mask + causal_masked_diff_xvec
audit("flow",
  paste(lines_of("R/s3gen.R", 17, 34), lines_of("R/s3gen.R", 799, 941),
    sep = "\n"),
  paste(readLines("R/yq_flow.R"), collapse = "\n"))

# TTS glue: create_voice_embedding + VE inference/windowing + xvector
# frontend + embed_ref + trim_silence + drop_invalid + trim_fade +
# generate tail
audit("tts glue",
  paste(lines_of("R/tts.R", 364, 455),
    lines_of("R/tts.R", 583, 700),
    lines_of("R/voice_encoder.R", 143, 268),
    lines_of("R/speaker_encoder.R", 430, 471),
    lines_of("R/s3gen.R", 954, 1056),
    lines_of("R/s3tokenizer.R", 658, 688),
    lines_of("R/audio_utils.R", 148, 182), sep = "\n"),
  paste(readLines("R/yq_tts.R"), collapse = "\n"))
