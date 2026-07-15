#!/usr/bin/env r
# Verify the anvl mel frontends against the torch fixtures. anvl/yunque only.
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_mel_fbank.R", envir = env)

fx <- readRDS("tools/fixtures/mel_fbank.rds")

report <- function(name, got, ref) {
  cat("==", name, "==\n")
  cat("got dims:", paste(dim(got), collapse = "x"), "\n")
  cat("ref dims:", paste(dim(ref), collapse = "x"), "\n")
  reltol <- max(abs(got - ref)) / max(abs(ref))
  cat(sprintf("reltol = %.3e   max|ref| = %.4e  max|got| = %.4e\n\n",
    reltol, max(abs(ref)), max(abs(got))))
  reltol
}

mel_got <- as.array(env$yq_compute_mel_spectrogram(fx$audio24))
r1 <- report("compute_mel_spectrogram", mel_got, fx$mel)

kal_got <- as.array(env$yq_kaldi_fbank(fx$audio16))
r2 <- report("kaldi_fbank", kal_got, fx$kaldi)

cat(sprintf("MAX reltol = %.3e  -> %s\n", max(r1, r2),
  if (max(r1, r2) < 1e-3) "PASS" else "FAIL"))
