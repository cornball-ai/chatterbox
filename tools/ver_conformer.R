#!/usr/bin/env r
# Verify the anvl conformer port against the torch fixture. Torch never loads.
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_common.R", envir = env)
sys.source("R/yq_conformer.R", envir = env)

fx <- readRDS("tools/fixtures/conformer.rds")

w <- env$yq_conformer_load_weights(fx$s3gen)
x <- anvl::nv_array(fx$x, dtype = "f32")
got <- as.array(env$yq_conformer(x, w))

ref <- fx$out
cat("ref dims:", paste(dim(ref), collapse = " x "), "\n")
cat("got dims:", paste(dim(got), collapse = " x "), "\n")
cat("dims match:", identical(dim(ref), dim(got)), "\n")

reltol <- max(abs(got - ref)) / max(abs(ref))
cat(sprintf("reltol = %.3e\n", reltol))
cat(sprintf("max|ref| = %.4f  max|got| = %.4f\n", max(abs(ref)), max(abs(got))))
