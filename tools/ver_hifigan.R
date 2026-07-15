#!/usr/bin/env r
# Verify the anvl HiFT/HiFiGAN port against the torch fixture.
# anvl/yunque only.
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_common.R", envir = env)
sys.source("R/yq_hifigan.R", envir = env)

fx <- readRDS("tools/fixtures/hifigan.rds")

w <- env$yq_hifigan_load_weights(fx$s3gen, prefix = "mel2wav.")
mel <- anvl::nv_array(fx$mel, dtype = "f32")

report <- function(label, got, ref) {
  cat(sprintf("%s: got %s ref %s reltol = %.3e\n", label,
    paste(dim(got), collapse = "x"), paste(dim(ref), collapse = "x"),
    max(abs(got - ref)) / max(abs(ref))))
}

# Check 1: f0 predictor forward (deterministic).
f0 <- as.array(env$yq_hifigan_f0(mel, w))
report("f0", f0, fx$f0)

# Check 3: source module voiced-path parity on the synthetic f0 sweep
# (reference m_source: sine bank -> harmonics linear -> tanh).
s3 <- as.array(env$.yq_hifigan_source(fx$f0_syn, w, fx$phase3, fx$z3))
report("sine_merge", s3, aperm(fx$sine_merge3, c(1L, 3L, 2L)))

# Check 2: full generator mel -> waveform with the reference RNG draws.
out <- env$yq_hifigan(mel, w, phase = fx$phase, noise = fx$z1)
report("source", as.array(out$source), fx$source)
report("audio", as.array(out$audio), fx$audio)
