#!/usr/bin/env r
# Verify the anvl CAMPPlus port against the torch fixture. anvl/yunque only.
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_campplus.R", envir = env)

fx <- readRDS("tools/fixtures/campplus.rds")

w <- env$yq_campplus_load_weights(fx$s3gen, prefix = "speaker_encoder.")
mels <- anvl::nv_array(fx$mel, dtype = "f32")
got <- as.array(env$yq_campplus(mels, w))
ref <- fx$out

cat("got dims:", paste(dim(got), collapse = "x"), "\n")
cat("ref dims:", paste(dim(ref), collapse = "x"), "\n")
reltol <- max(abs(got - ref)) / max(abs(ref))
cat(sprintf("reltol = %.3e\n", reltol))
