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
