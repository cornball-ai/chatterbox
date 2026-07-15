#!/usr/bin/env Rscript --vanilla
# Fixture for the windowed-sinc resampler. Torch only (CPU). No weights.
library(chatterbox)

set.seed(1234)
audio <- rnorm(24000) * 0.1 # 1 s at 24 kHz

cases <- list(
  c(24000L, 16000L), # ref audio -> 16 kHz frontends
  c(16000L, 24000L), # 16 kHz -> S3Gen rate
  c(44100L, 16000L) # arbitrary input rate
)

out <- lapply(cases, function(cs) {
  torch::with_no_grad({
    as.numeric(chatterbox:::sinc_resample(audio, cs[1], cs[2]))
  })
})

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(audio = audio, cases = cases, out = out),
  "tools/fixtures/resample.rds")
cat("resample fixture:", paste(sapply(out, length), collapse = ", "),
  "samples\n")
