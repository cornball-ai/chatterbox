#!/usr/bin/env Rscript --vanilla
# Fixtures for the two mel frontends. Torch only (CPU). No weights.
library(chatterbox)

paths <- chatterbox:::get_model_paths()

set.seed(1234)
# 24 kHz clip for the S3Gen reference mel (0.5 s)
audio24 <- rnorm(12000) * 0.1
# 16 kHz clip for the Kaldi CAMPPlus fbank (0.5 s)
audio16 <- rnorm(8000) * 0.1

mel <- torch::with_no_grad({
  chatterbox:::compute_mel_spectrogram(audio24)
})
kal <- torch::with_no_grad({
  chatterbox:::kaldi_fbank(audio16)
})
ve <- torch::with_no_grad({
  chatterbox:::compute_mel_spectrogram_ve(audio16)
})

mel_a <- as.array(mel$cpu())
kal_a <- as.array(kal$cpu())
ve_a <- as.array(ve$cpu())

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(audio24 = audio24, audio16 = audio16,
  mel = mel_a, kaldi = kal_a, ve = ve_a, s3gen = paths$s3gen),
  "tools/fixtures/mel_fbank.rds")
cat("mel dims", paste(dim(mel_a), collapse = "x"),
  " kaldi dims", paste(dim(kal_a), collapse = "x"),
  " ve dims", paste(dim(ve_a), collapse = "x"), "\n")
