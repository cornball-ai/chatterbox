#!/usr/bin/env r
# Verify the anvl voice-embedding stage against the torch fixture.
# anvl/yunque only (no torch in this process).
suppressMessages({
  library(anvl)
  library(yunque)
})

e <- new.env()
for (f in c("yq_common.R", "yq_resample.R", "yq_mel_fbank.R",
  "yq_voice_encoder.R", "yq_s3tokenizer.R", "yq_campplus.R", "yq_tts.R")) {
  sys.source(file.path("R", f), envir = e)
}

fx <- readRDS("tools/fixtures/voice_embedding.rds")
w <- list(
  ve = e$yq_ve_load_weights(fx$ve_path),
  campplus = e$yq_campplus_load_weights(fx$s3gen_path),
  s3tok = e$yq_s3tokenizer_load_weights(fx$s3gen_path)
)
res <- e$yq_voice_embedding(fx$samples, fx$sr, w)

reltol <- function(a, b) max(abs(a - b)) / max(abs(b))
ok <- TRUE
num <- function(name, got, ref, tol = 1e-3) {
  r <- reltol(got, ref)
  pass <- identical(dim(got), dim(ref)) && r < tol
  ok <<- ok && pass
  cat(sprintf("[%-18s] %s  reltol %.3e  dim %s\n", name,
    if (pass) "PASS" else "FAIL", r, paste(dim(got), collapse = "x")))
}
tok <- function(name, got, ref, min_rate = 0.99) {
  same_dim <- identical(dim(got), dim(ref))
  rate <- if (same_dim) mean(got == ref) else 0
  pass <- same_dim && rate >= min_rate
  ok <<- ok && pass
  cat(sprintf("[%-18s] %s  exact-match %.4f  n %d  dim %s\n", name,
    if (pass) "PASS" else "FAIL", rate, length(ref),
    paste(dim(got), collapse = "x")))
}

num("ve_embedding", res$ve_embedding, fx$ve_embedding)
tok("cond_prompt_tokens", res$cond_prompt_speech_tokens, fx$cond_prompt_tokens)
tok("prompt_token", res$ref_dict$prompt_token, fx$prompt_token)
num("prompt_feat", res$ref_dict$prompt_feat, fx$prompt_feat)
num("xvector", res$ref_dict$embedding, fx$xvector)

quit(status = if (ok) 0L else 1L)
