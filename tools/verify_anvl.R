#!/usr/bin/env r
# Verify the anvl/yunque chatterbox port against the torch fixtures. Anvl-only
# process (no library(chatterbox), so torch never loads). CPU by default.
suppressMessages({
  library(anvl)
  library(yunque)
})

e <- new.env()
sys.source("R/yq_voice_encoder.R", envir = e)

reltol <- function(a, b) max(abs(a - b)) / max(abs(b))
ok <- TRUE
report <- function(name, got, ref, tol = 1e-3) {
  r <- reltol(got, ref)
  pass <- identical(dim(got), dim(ref)) && r < tol
  ok <<- ok && pass
  cat(sprintf("[%-14s] %s  reltol %.2e  dim %s\n", name,
    if (pass) "PASS" else "FAIL", r, paste(dim(got), collapse = "x")))
}

# --- voice encoder ---
fix <- readRDS("tools/fixtures/voice_encoder.rds")
w <- e$yq_ve_load_weights(fix$ve_path)
emb <- as.array(e$yq_voice_encoder(nv_array(fix$mels, dtype = "f32"), w))
report("voice_encoder", emb, fix$emb)

quit(status = if (ok) 0L else 1L)
