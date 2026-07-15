#!/usr/bin/env Rscript --vanilla
# Generate the CFM flow reference fixtures with the torch impl. CPU only.
# Check 1: cfm_estimator forward alone (real flow.decoder.estimator. weights).
# Check 2: full flow inference (token ids -> mel) through
#   causal_masked_diff_xvec with the decoder's rand_noise buffer overwritten
#   by seeded noise (the reference slices that buffer for its initial z), so
#   the anvl port can be fed the identical noise explicitly.
suppressMessages(library(chatterbox))

paths <- chatterbox:::get_model_paths()
sd <- safetensors::safe_load_file(paths$s3gen, framework = "torch",
  device = "cpu")

ft <- function(x) torch::torch_tensor(x, dtype = torch::torch_float())

# ---------------------------------------------------------------------------
# Check 1: estimator forward alone
# ---------------------------------------------------------------------------
est <- chatterbox:::cfm_estimator(in_channels = 320L, out_channels = 80L,
  hidden_dim = 256L, num_mid_blocks = 12L, num_transformer_blocks = 4L,
  meanflow = FALSE)
n_loaded <- torch::with_no_grad({
  chatterbox:::load_cfm_estimator_weights(est, sd,
    prefix = "flow.decoder.estimator.")
})
cat("estimator params loaded:", n_loaded, "\n")
est$eval()

set.seed(1234)
B <- 2L
TT <- 32L
x1 <- ft(array(rnorm(B * 80L * TT), c(B, 80L, TT)))
mask1 <- ft(array(1, c(B, 1L, TT)))
mu1 <- ft(array(rnorm(B * 80L * TT), c(B, 80L, TT)))
# f32-exact timesteps (dyadic rationals) so host-side sinusoids see the
# same t bits torch does
t1 <- c(0.375, 0.8125)
spks1 <- ft(matrix(rnorm(B * 80L), B, 80L))
cond1 <- ft(array(rnorm(B * 80L * TT), c(B, 80L, TT)))

out1 <- torch::with_no_grad({
  est$forward(x1, mask1, mu1, ft(t1), spks1, cond1)$cpu()
})
cat("check1 out dims:", paste(dim(out1), collapse = " x "), "\n")

# ---------------------------------------------------------------------------
# Check 2: full flow inference (token ids -> mel)
# ---------------------------------------------------------------------------
flow <- chatterbox:::causal_masked_diff_xvec(vocab_size = 6561,
  input_size = 512, output_size = 80, spk_embed_dim = 192,
  input_frame_rate = 25, token_mel_ratio = 2, meanflow = FALSE)

torch::with_no_grad({
  flow$input_embedding$weight$copy_(sd[["flow.input_embedding.weight"]])
  flow$spk_embed_affine_layer$weight$copy_(
    sd[["flow.spk_embed_affine_layer.weight"]])
  flow$spk_embed_affine_layer$bias$copy_(
    sd[["flow.spk_embed_affine_layer.bias"]])
  flow$encoder_proj$weight$copy_(sd[["flow.encoder_proj.weight"]])
  flow$encoder_proj$bias$copy_(sd[["flow.encoder_proj.bias"]])
  chatterbox:::load_conformer_encoder_weights(flow$encoder, sd,
    prefix = "flow.encoder.")
  chatterbox:::load_cfm_estimator_weights(flow$decoder$estimator, sd,
    prefix = "flow.decoder.estimator.")
})
flow$eval()

# Control the stochastic boundary: overwrite the decoder's pre-generated
# noise buffer (torch_randn at init) with seeded noise, and save the slice
# the solver will consume.
noise_full <- array(rnorm(1L * 80L * 15000L), c(1L, 80L, 15000L))
torch::with_no_grad({
  flow$decoder$rand_noise$copy_(ft(noise_full))
})

n_speech <- 24L
n_prompt <- 16L
speech_tokens <- sample(0:6560, n_speech, replace = TRUE)
prompt_tokens <- sample(0:6560, n_prompt, replace = TRUE)
mel_len1 <- 2L * n_prompt
prompt_feat_t <- ft(array(rnorm(1L * mel_len1 * 80L), c(1L, mel_len1, 80L)))
embedding_t <- ft(matrix(rnorm(192L), 1L, 192L))
mel_total <- 2L * (n_speech + n_prompt)

tok_t <- torch::torch_tensor(matrix(speech_tokens, 1L),
  dtype = torch::torch_long())
ptok_t <- torch::torch_tensor(matrix(prompt_tokens, 1L),
  dtype = torch::torch_long())
tlen_t <- torch::torch_tensor(n_speech, dtype = torch::torch_long())
ptlen_t <- torch::torch_tensor(n_prompt, dtype = torch::torch_long())

res <- torch::with_no_grad({
  flow$forward(token = tok_t, token_len = tlen_t, prompt_token = ptok_t,
    prompt_token_len = ptlen_t, prompt_feat = prompt_feat_t,
    prompt_feat_len = NULL, embedding = embedding_t, finalize = TRUE,
    traced = FALSE, n_timesteps = NULL)
})
out2 <- as.array(res[[1]]$cpu())
cat("check2 out dims:", paste(dim(out2), collapse = " x "), "\n")

# Debug intermediates (real submodules, for triage if check 2 diverges).
dbg <- torch::with_no_grad({
  emb_n <- torch::nnf_normalize(embedding_t, dim = 2)
  spks_dbg <- flow$spk_embed_affine_layer$forward(emb_n)
  tok_full <- torch::torch_cat(list(ptok_t, tok_t), dim = 2)
  tok_emb <- flow$input_embedding$forward(tok_full$add(1L))
  enc <- flow$encoder$forward(tok_emb,
    torch::torch_tensor(n_speech + n_prompt, dtype = torch::torch_long()))
  mu_dbg <- flow$encoder_proj$forward(enc[[1]])
  list(spks = as.array(spks_dbg$cpu()), mu = as.array(mu_dbg$cpu()))
})

# Exact f32 values of the reference's internal time schedule and sinusoid
# frequency table (extracted with the same torch ops the modules run), so
# the port's host-side schedule can be checked bit-for-bit.
t_span_t <- torch::torch_linspace(0, 1, 11L, dtype = torch::torch_float())
t_span_t <- 1 - torch::torch_cos(t_span_t * 0.5 * pi)
t_span <- as.numeric(as.array(t_span_t))
freqs_t <- torch::torch_exp(
  torch::torch_arange(0, 159, dtype = torch::torch_float()) *
  (-log(10000) / 159))
sinu_freqs <- as.numeric(as.array(freqs_t))

dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(
  s3gen = paths$s3gen,
  # check 1 (inputs saved from the f32 tensors actually fed to torch)
  x1 = as.array(x1), mu1 = as.array(mu1), spks1 = as.array(spks1),
  cond1 = as.array(cond1), t1 = t1, out1 = as.array(out1),
  # check 2
  speech_tokens = speech_tokens, prompt_tokens = prompt_tokens,
  prompt_feat = as.array(prompt_feat_t), embedding = as.array(embedding_t),
  noise = as.array(flow$decoder$rand_noise[, , 1:mel_total]$cpu()),
  out2 = out2,
  spks_dbg = dbg$spks, mu_dbg = dbg$mu,
  t_span = t_span, sinu_freqs = sinu_freqs
), "tools/fixtures/cfm.rds")
cat("saved tools/fixtures/cfm.rds\n")
