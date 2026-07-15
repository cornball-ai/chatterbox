#!/usr/bin/env r
# Verify the anvl CFM flow port against the torch fixture. Torch never loads.
# Run: PJRT_PLATFORM=cpu r tools/ver_cfm.R
library(anvl)
library(yunque)

env <- new.env()
sys.source("R/yq_common.R", envir = env)
sys.source("R/yq_conformer.R", envir = env)
sys.source("R/yq_cfm.R", envir = env)
sys.source("R/yq_flow.R", envir = env)

fx <- readRDS("tools/fixtures/cfm.rds")
nv <- function(x) anvl::nv_array(x, dtype = "f32")
report <- function(label, got, ref) {
  reltol <- max(abs(got - ref)) / max(abs(ref))
  cat(sprintf("%s: dims got %s / ref %s match=%s  reltol = %.3e\n", label,
    paste(dim(got), collapse = "x"), paste(dim(ref), collapse = "x"),
    identical(dim(got), dim(ref)), reltol))
  reltol
}

# ---------------------------------------------------------------------------
# Check 1: estimator forward alone
# ---------------------------------------------------------------------------
w <- env$yq_cfm_load_weights(fx$s3gen)
cat(sprintf("sinu_freqs host vs torch: max abs diff = %.3e\n",
  max(abs(w$sinu_freqs - fx$sinu_freqs))))
w$sinu_freqs <- fx$sinu_freqs

mask1 <- nv(array(1, c(dim(fx$x1)[1L], 1L, dim(fx$x1)[3L])))
got1 <- as.array(env$yq_cfm_estimator(nv(fx$x1), mask1, nv(fx$mu1), fx$t1,
  nv(fx$spks1), nv(fx$cond1), w))
report("check1 estimator (ones mask)", got1, fx$out1)

got1n <- as.array(env$yq_cfm_estimator(nv(fx$x1), NULL, nv(fx$mu1), fx$t1,
  nv(fx$spks1), nv(fx$cond1), w))
cat(sprintf("check1 mask=NULL vs ones mask: max abs diff = %.3e\n",
  max(abs(got1n - got1))))

# ---------------------------------------------------------------------------
# Check 2: full flow inference (token ids -> mel)
# ---------------------------------------------------------------------------
wf <- env$yq_flow_load_weights(fx$s3gen)

# Triage intermediates (host speaker path + encoder side).
e <- as.numeric(fx$embedding)
e <- e / max(sqrt(sum(e^2)), 1e-12)
spks_got <- as.array(yunque::linear(nv(matrix(e, 1L)), wf$spk_w, wf$spk_b))
report("  [dbg] spk_embed_affine", spks_got, fx$spks_dbg)
tokens <- c(fx$prompt_tokens, fx$speech_tokens)
emb <- wf$input_embedding[tokens + 1L, , drop = FALSE]
h <- env$yq_conformer(nv(array(emb, c(1L, length(tokens), ncol(emb)))),
  wf$encoder)
mu_got <- as.array(yunque::linear(h, wf$proj_w, wf$proj_b))
report("  [dbg] encoder + proj (mu)", mu_got, fx$mu_dbg)

# Strict parity: reference-exact f32 schedule + sinusoid table.
wf$cfm$sinu_freqs <- fx$sinu_freqs
got2 <- as.array(env$yq_flow_inference(fx$speech_tokens, fx$prompt_tokens,
  fx$prompt_feat, fx$embedding, wf, fx$noise, n_timesteps = 10L,
  t_span = fx$t_span))
report("check2 flow tokens->mel (torch t_span)", got2, fx$out2)

# Real-usage config: host-computed schedule and frequency table.
wf$cfm$sinu_freqs <- env$.yq_cfm_sinu_freqs(160L)
cat(sprintf("t_span host vs torch: max abs diff = %.3e\n",
  max(abs(env$.yq_cfm_t_span(10L) - fx$t_span))))
got2h <- as.array(env$yq_flow_inference(fx$speech_tokens, fx$prompt_tokens,
  fx$prompt_feat, fx$embedding, wf, fx$noise, n_timesteps = 10L))
report("check2 flow tokens->mel (host schedule)", got2h, fx$out2)
