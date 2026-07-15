#!/usr/bin/env Rscript --vanilla
# Fixture generator for the CAMPPlus speaker encoder. Torch only (CPU).
library(chatterbox)

paths <- chatterbox:::get_model_paths()

model <- chatterbox:::campplus(feat_dim = 80, embedding_size = 192)
sd <- safetensors::safe_load_file(paths$s3gen, framework = "torch",
  device = "cpu")
torch::with_no_grad({
  model <- chatterbox:::load_campplus_weights(model, sd,
    prefix = "speaker_encoder.")
})
model$eval()
model$to(device = "cpu", dtype = torch::torch_float())

set.seed(1234)
B <- 1L
T <- 256L
F <- 80L
mel <- array(rnorm(B * T * F), dim = c(B, T, F))
mel_t <- torch::torch_tensor(mel, dtype = torch::torch_float())

out <- torch::with_no_grad({
  model$forward(mel_t)
})

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(mel = mel, out = as.array(out$cpu()), s3gen = paths$s3gen),
  "tools/fixtures/campplus.rds")
cat("saved fixture: out dims", paste(dim(as.array(out$cpu())), collapse = "x"),
  "\n")
