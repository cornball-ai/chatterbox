#!/usr/bin/env Rscript
# Torch reference fixtures for the anvl/yunque chatterbox port. Run with the
# torch chatterbox installed. CPU-only. Writes tools/fixtures/*.rds. Kept
# separate from the anvl verify so torch and anvl never co-load.
suppressMessages(library(chatterbox))

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
set.seed(1234)
paths <- chatterbox:::get_model_paths()

# --- voice encoder (3-layer LSTM speaker embedding) ---
cfg <- chatterbox:::voice_encoder_config()
ve <- chatterbox:::voice_encoder(cfg)
sd <- safetensors::safe_load_file(paths$ve, framework = "torch", device = "cpu")
chatterbox:::load_voice_encoder_weights(ve, sd)
ve$eval()
n_frames <- 120L
mels <- array(rnorm(1 * n_frames * cfg$num_mels), c(1L, n_frames, cfg$num_mels))
emb <- as.array(torch::with_no_grad(
  ve$forward(torch::torch_tensor(mels, dtype = torch::torch_float()))))
saveRDS(list(mels = mels, emb = emb, ve_path = paths$ve),
  "tools/fixtures/voice_encoder.rds")
cat(sprintf("voice_encoder fixture: emb %s\n", paste(dim(emb), collapse = "x")))
