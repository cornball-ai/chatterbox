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
  chatterbox:::compute_ve_mel(audio16)
})
# S3 tokenizer whisper-style log-mel (filters/window built as the module does)
s3fb <- chatterbox:::create_mel_filterbank(sr = 16000, n_fft = 400L,
  n_mels = 128L, fmin = 0, fmax = 8000)
s3m <- torch::with_no_grad({
  chatterbox:::s3_log_mel_spectrogram(matrix(audio16, nrow = 1L),
    torch::torch_tensor(s3fb, dtype = torch::torch_float32()),
    torch::torch_hann_window(400L))
})

mel_a <- as.array(mel$cpu())
kal_a <- as.array(kal$cpu())
ve_a <- as.array(ve$cpu())
s3m_a <- as.array(s3m$cpu())

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(audio24 = audio24, audio16 = audio16,
  mel = mel_a, kaldi = kal_a, ve = ve_a, s3mel = s3m_a, s3gen = paths$s3gen),
  "tools/fixtures/mel_fbank.rds")
cat("mel dims", paste(dim(mel_a), collapse = "x"),
  " kaldi dims", paste(dim(kal_a), collapse = "x"),
  " ve dims", paste(dim(ve_a), collapse = "x"),
  " s3mel dims", paste(dim(s3m_a), collapse = "x"), "\n")
