#!/usr/bin/env Rscript --vanilla
# Torch reference fixture for the S3 tokenizer encoder + FSQ quantizer.
# CPU only (GPU is busy). Feeds a deterministic random log-mel through the
# torch s3_audio_encoder and FSQ quantizer, saves hidden + tokens.
library(chatterbox)

paths <- chatterbox:::get_model_paths()
sd <- safetensors::safe_load_file(paths$s3gen, framework = "torch",
  device = "cpu")

model <- chatterbox:::s3_tokenizer()
model <- chatterbox:::load_s3tokenizer_weights(model, sd, prefix = "tokenizer.")
model$eval()
model$to(device = "cpu", dtype = torch::torch_float())

set.seed(1234)
Tt <- 128L
mel <- torch::torch_randn(1L, 128L, Tt, dtype = torch::torch_float())
mel_len <- torch::torch_tensor(Tt, dtype = torch::torch_long())$unsqueeze(1L)

res <- torch::with_no_grad({
  enc <- model$encoder$forward(mel, mel_len)
  hidden <- enc$hidden
  tokens <- model$quantizer$encode(hidden)
  list(hidden = as.array(hidden$cpu()),
       tokens = as.array(tokens$cpu()))
})

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(mel = as.array(mel$cpu()), mel_len = Tt,
  hidden = res$hidden, tokens = res$tokens, s3gen = paths$s3gen),
  "tools/fixtures/s3tokenizer.rds")

cat("hidden dims:", paste(dim(res$hidden), collapse = "x"), "\n")
cat("tokens dims:", paste(dim(res$tokens), collapse = "x"), "\n")
cat("token range:", range(res$tokens), "\n")
