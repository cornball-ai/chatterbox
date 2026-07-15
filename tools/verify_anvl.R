#!/usr/bin/env r
# Verify the anvl/yunque chatterbox port against the torch fixtures. Anvl-only
# process (no library(chatterbox), so torch never loads). CPU by default.
suppressMessages({
  library(anvl)
  library(yunque)
})

e <- new.env()
sys.source("R/yq_voice_encoder.R", envir = e)
sys.source("R/yq_llama.R", envir = e)
sys.source("R/yq_t3_cond.R", envir = e)

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

# --- T3 Llama backbone ---
fixl <- readRDS("tools/fixtures/llama.rds")
wl <- e$yq_llama_load_weights(fixl$t3_path)
out <- as.array(e$yq_llama(nv_array(fixl$embeds, dtype = "f32"), wl))
report("llama", out, fixl$out)

# --- T3 conditioning encoder ---
fixc <- readRDS("tools/fixtures/t3_cond.rds")
wc <- e$yq_t3_cond_load_weights(fixc$t3_path)
cout <- as.array(e$yq_t3_cond_enc(nv_array(fixc$spk, dtype = "f32"),
  nv_array(fixc$prompt, dtype = "f32"), fixc$emo, wc))
report("t3_cond", cout, fixc$out)

quit(status = if (ok) 0L else 1L)
