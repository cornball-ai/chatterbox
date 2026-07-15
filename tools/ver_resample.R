#!/usr/bin/env r
# Verify the anvl sinc resampler against the torch fixture. anvl/yunque only.
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_resample.R", envir = env)

fx <- readRDS("tools/fixtures/resample.rds")

ok <- TRUE
for (i in seq_along(fx$cases)) {
  cs <- fx$cases[[i]]
  got <- env$yq_resample(fx$audio, cs[1], cs[2])
  ref <- fx$out[[i]]
  reltol <- max(abs(got - ref)) / max(abs(ref))
  pass <- length(got) == length(ref) && reltol < 1e-3
  ok <- ok && pass
  cat(sprintf("[%d -> %d] %s  reltol %.3e  n %d (ref %d)\n",
    cs[1], cs[2], if (pass) "PASS" else "FAIL", reltol,
    length(got), length(ref)))
}
quit(status = if (ok) 0L else 1L)
