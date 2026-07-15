#!/usr/bin/env r
# anvl/yunque parity check for the S3 tokenizer. Torch is never loaded.
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_llama.R", envir = env) # provides .yq_heads
sys.source("R/yq_s3tokenizer.R", envir = env)

fx <- readRDS("tools/fixtures/s3tokenizer.rds")
w <- env$yq_s3tokenizer_load_weights(fx$s3gen)

mel <- anvl::nv_array(fx$mel, dtype = "f32")
res <- env$yq_s3tokenizer(mel, w)

got <- as.array(res$hidden)
ref <- fx$hidden
reltol <- max(abs(got - ref)) / max(abs(ref))

cat("hidden dims got:", paste(dim(got), collapse = "x"),
  " ref:", paste(dim(ref), collapse = "x"), "\n")
cat("dims match:", identical(dim(got), dim(ref)), "\n")
cat("hidden reltol:", format(reltol, scientific = TRUE), "\n")

tok_ref <- as.integer(fx$tokens)
tok_got <- as.integer(res$tokens)
cat("token exact-match rate:", mean(tok_got == tok_ref), "\n")
cat("n tokens:", length(tok_got), " mismatches:", sum(tok_got != tok_ref), "\n")
