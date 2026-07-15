#!/usr/bin/env Rscript --vanilla
# Generate the conformer (flow.encoder) reference fixture with the torch impl.
# CPU only -- the GPU is busy and torch/anvl clash on CUDA.
suppressMessages(library(chatterbox))

paths <- chatterbox:::get_model_paths()
sd <- safetensors::safe_load_file(paths$s3gen, framework = "torch",
  device = "cpu")

prefix <- "flow.encoder."
keys <- grep(paste0("^", prefix), names(sd), value = TRUE)
enc_sd <- sd[keys]
names(enc_sd) <- sub(paste0("^", prefix), "", keys)

module <- chatterbox:::upsample_conformer_encoder_full(
  input_size = 512, output_size = 512, num_blocks = 6, num_up_blocks = 4,
  n_head = 8, n_ffn = 2048, dropout_rate = 0.0, pre_lookahead_len = 3)
module <- chatterbox:::load_conformer_encoder_weights(module, sd, prefix = prefix)
module$eval()
module$to(device = "cpu", dtype = torch::torch_float())

set.seed(1234)
T <- 12L
x_arr <- array(rnorm(1L * T * 512L), c(1L, T, 512L))
x <- torch::torch_tensor(x_arr, dtype = torch::torch_float())
x_lens <- torch::torch_tensor(as.integer(T), dtype = torch::torch_long())

out <- torch::with_no_grad({
  res <- module$forward(x, x_lens)
  res[[1]]$cpu()
})
out_arr <- as.array(out)

cat("out dims:", paste(dim(out_arr), collapse = " x "), "\n")

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(x = x_arr, T = T, out = out_arr, s3gen = paths$s3gen),
  "tools/fixtures/conformer.rds")
cat("saved tools/fixtures/conformer.rds\n")
