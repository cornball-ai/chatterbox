#!/usr/bin/env Rscript
# Check eval mode and dropout behavior

library(torch)
devtools::load_all("/home/troy/chatterbox")

# Create encoder
encoder <- upsample_conformer_encoder_full()
encoder$eval()

cat("=== Check training mode ===\n")
cat("encoder$training:", encoder$training, "\n")
cat("encoders[[1]]$training:", encoder$encoders[[1]]$training, "\n")
cat("encoders[[1]]$dropout$training:", encoder$encoders[[1]]$dropout$training, "\n")
cat("encoders[[1]]$self_attn$dropout$training:", encoder$encoders[[1]]$self_attn$dropout$training, "\n")
cat("embed$pos_enc$dropout$training:", encoder$embed$pos_enc$dropout$training, "\n")

# Test dropout behavior in eval mode
cat("\n=== Test dropout in eval mode ===\n")
x <- torch::torch_randn(c(1, 32, 512))

dropout <- torch::nn_dropout(0.1)
dropout$eval()

y1 <- dropout$forward(x)
y2 <- dropout$forward(x)

diff <- (y1 - y2)$abs()$max()$item()
cat("Dropout output diff (should be 0 in eval):", diff, "\n")

identity_diff <- (y1 - x)$abs()$max()$item()
cat("Dropout vs identity diff (should be 0 in eval):", identity_diff, "\n")

cat("\n=== Done ===\n")

