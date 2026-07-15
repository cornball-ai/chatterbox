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
