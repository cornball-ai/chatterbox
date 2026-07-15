# anvl/yunque port of the S3Gen CFM decoder (flow.decoder): the
# ConditionalDecoder estimator (UNet-ish stack of causal resnet blocks and
# transformer blocks) and the CausalConditionalCFM Euler solver with
# classifier-free guidance. Torch-free; yq_ marks the port. Batch-1 /
# full-length (unpadded) inference path -- the (all-ones) feature mask is
# applied where the reference applies it, but attention always runs
# unmasked, so genuinely padded batches are not supported.

# .yq_f32 (float32 rounding emulation for the reference's torch-f32
# scalar chains) comes from yq_common.R (source it first when standalone).

# Sinusoidal timestep frequency table: exp(arange(half) * -log(1e4)/(half-1)),
# f32-rounded like the reference's torch chain.
.yq_cfm_sinu_freqs <- function(half_dim = 160L) {
  .yq_f32(exp(.yq_f32((0:(half_dim - 1L)) * (-log(10000) / (half_dim - 1L)))))
}

# Timestep embedding: host sinusoid (scale 1000, sin|cos -> [B, 320]) then
# Linear/SiLU/Linear -> [B, 1024]. t is a host numeric vector, length B.
.yq_cfm_time_emb <- function(t, w) {
  ang <- .yq_f32(outer(.yq_f32(1000 * t), w$sinu_freqs))
  emb <- anvl::nv_array(cbind(sin(ang), cos(ang)), dtype = "f32")
  h <- yunque::silu(yunque::linear(emb, w$time_w1, w$time_b1))
  yunque::linear(h, w$time_w2, w$time_b2)
}

# x * mask with mask [B, 1, T] broadcast over channels; NULL = identity
# (the batch-1 inference mask is all ones).
.yq_cfm_mask_mul <- function(x, mask) {
  if (is.null(mask)) {
    return(x)
  }
  x * anvl::nv_broadcast_to(mask, anvl::shape(x))
}

# Causal Conv1d: left-pad (k-1)*dilation, then valid conv. x [B, C, T].
.yq_cfm_causal_conv <- function(x, w, b, dilation = 1L) {
  k <- anvl::shape(w)[3L]
  pad <- (k - 1L) * dilation
  if (pad > 0L) {
    x <- anvl::nv_pad(x, 0, c(0L, 0L, pad), c(0L, 0L, 0L))
  }
  yunque::conv1d(x, w, b, dilation = dilation)
}

# CausalBlock1D: causal conv (k=3) -> LayerNorm over channels -> Mish,
# masked at input and output. [B, C, T].
.yq_cfm_block1d <- function(x, mask, blk) {
  h <- .yq_cfm_causal_conv(.yq_cfm_mask_mul(x, mask), blk$conv_w, blk$conv_b)
  h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  h <- yunque::layer_norm(h, blk$ln_w, blk$ln_b, eps = 1e-5)
  h <- yunque::mish(anvl::nv_transpose(h, c(1L, 3L, 2L)))
  .yq_cfm_mask_mul(h, mask)
}

# CausalResnetBlock1D: two causal blocks with a per-channel timestep shift
# after the first, plus a 1x1-conv residual.
.yq_cfm_resnet <- function(x, mask, t_emb, r) {
  h <- .yq_cfm_block1d(x, mask, r$block1)
  shift <- yunque::linear(yunque::mish(t_emb), r$mlp_w, r$mlp_b) # [B, C]
  sh <- anvl::shape(h)
  h <- h + anvl::nv_broadcast_to(
    anvl::nv_reshape(shift, c(sh[1L], sh[2L], 1L)), sh)
  h <- .yq_cfm_block1d(h, mask, r$block2)
  h + yunque::conv1d(.yq_cfm_mask_mul(x, mask), r$res_w, r$res_b)
}

# BasicTransformerBlock: pre-norm SDPA (8 heads x 64 -> inner 512, exact-GELU
# FFN 256 -> 1024 -> 256), both residual. x [B, T, 256], unmasked attention.
.yq_cfm_tfm_block <- function(x, blk, batch, n_head = 8L, head_dim = 64L) {
  seq_len <- anvl::shape(x)[2L]
  xn <- yunque::layer_norm(x, blk$norm1_w, blk$norm1_b, eps = 1e-5)
  q <- .yq_heads(yunque::linear(xn, blk$q_w), batch, n_head, head_dim)
  k <- .yq_heads(yunque::linear(xn, blk$k_w), batch, n_head, head_dim)
  v <- .yq_heads(yunque::linear(xn, blk$v_w), batch, n_head, head_dim)
  a <- yunque::sdpa(q, k, v)
  a <- anvl::nv_reshape(anvl::nv_transpose(a, c(1L, 3L, 2L, 4L)),
    c(batch, seq_len, n_head * head_dim))
  x <- x + yunque::linear(a, blk$out_w, blk$out_b)

  xn <- yunque::layer_norm(x, blk$norm3_w, blk$norm3_b, eps = 1e-5)
  ff <- yunque::linear(
    yunque::gelu(yunque::linear(xn, blk$ff1_w, blk$ff1_b),
      approximate = "none"),
    blk$ff2_w, blk$ff2_b)
  x + ff
}

#' Load CFM estimator (ConditionalDecoder) weights (anvl)
#'
#' @param path Path to s3gen.safetensors.
#' @param prefix Key prefix (default \code{"flow.decoder.estimator."}).
#' @param num_mid_blocks Mid blocks (default 12).
#' @param num_transformer_blocks Transformer blocks per stack (default 4).
#' @return List of weights for \code{\link{yq_cfm_estimator}}.
#' @export
yq_cfm_load_weights <- function(path, prefix = "flow.decoder.estimator.",
                                num_mid_blocks = 12L,
                                num_transformer_blocks = 4L) {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  block1d <- function(pre) list(
    conv_w = nv(paste0(pre, "block.0.weight")),
    conv_b = nv(paste0(pre, "block.0.bias")),
    ln_w = nv(paste0(pre, "block.2.weight")),
    ln_b = nv(paste0(pre, "block.2.bias"))
  )
  resnet <- function(pre) list(
    block1 = block1d(paste0(pre, "block1.")),
    block2 = block1d(paste0(pre, "block2.")),
    mlp_w = nv(paste0(pre, "mlp.1.weight"), TRUE),
    mlp_b = nv(paste0(pre, "mlp.1.bias")),
    res_w = nv(paste0(pre, "res_conv.weight")),
    res_b = nv(paste0(pre, "res_conv.bias"))
  )
  tfm <- function(pre) list(
    norm1_w = nv(paste0(pre, "norm1.weight")),
    norm1_b = nv(paste0(pre, "norm1.bias")),
    q_w = nv(paste0(pre, "attn1.to_q.weight"), TRUE),
    k_w = nv(paste0(pre, "attn1.to_k.weight"), TRUE),
    v_w = nv(paste0(pre, "attn1.to_v.weight"), TRUE),
    out_w = nv(paste0(pre, "attn1.to_out.0.weight"), TRUE),
    out_b = nv(paste0(pre, "attn1.to_out.0.bias")),
    norm3_w = nv(paste0(pre, "norm3.weight")),
    norm3_b = nv(paste0(pre, "norm3.bias")),
    ff1_w = nv(paste0(pre, "ff.net.0.proj.weight"), TRUE),
    ff1_b = nv(paste0(pre, "ff.net.0.proj.bias")),
    ff2_w = nv(paste0(pre, "ff.net.2.weight"), TRUE),
    ff2_b = nv(paste0(pre, "ff.net.2.bias"))
  )
  tfm_stack <- function(pre) lapply(seq_len(num_transformer_blocks) - 1L,
    function(i) tfm(sprintf("%s%d.", pre, i)))
  list(
    sinu_freqs = .yq_cfm_sinu_freqs(160L),
    time_w1 = nv("time_mlp.linear_1.weight", TRUE),
    time_b1 = nv("time_mlp.linear_1.bias"),
    time_w2 = nv("time_mlp.linear_2.weight", TRUE),
    time_b2 = nv("time_mlp.linear_2.bias"),
    down_resnet = resnet("down_blocks.0.0."),
    down_tfm = tfm_stack("down_blocks.0.1."),
    down_conv_w = nv("down_blocks.0.2.weight"),
    down_conv_b = nv("down_blocks.0.2.bias"),
    mid = lapply(seq_len(num_mid_blocks) - 1L, function(i) list(
      resnet = resnet(sprintf("mid_blocks.%d.0.", i)),
      tfm = tfm_stack(sprintf("mid_blocks.%d.1.", i))
    )),
    up_resnet = resnet("up_blocks.0.0."),
    up_tfm = tfm_stack("up_blocks.0.1."),
    up_conv_w = nv("up_blocks.0.2.weight"),
    up_conv_b = nv("up_blocks.0.2.bias"),
    final_block = block1d("final_block."),
    final_proj_w = nv("final_proj.weight"),
    final_proj_b = nv("final_proj.bias")
  )
}

#' CFM estimator (ConditionalDecoder) forward (anvl)
#'
#' Torch-free port of the S3Gen CFM estimator: inputs pack to
#' \code{[B, 320, T]} (x + mu + speaker + cond), one down stage
#' (resnet + 4 transformer blocks + causal conv, skip saved), 12 mid stages
#' (resnet + 4 transformer blocks), one up stage (skip concat -> resnet ->
#' 4 transformer blocks -> causal conv), final causal block + 1x1 projection.
#' Full-length (unpadded) sequences only: \code{mask} multiplies features at
#' the reference's mask points but attention runs unmasked, so pass all-ones
#' (or NULL, which skips the identity multiplies).
#'
#' @param x AnvlArray \code{[B, 80, T]} noisy sample.
#' @param mask AnvlArray \code{[B, 1, T]} of ones, or NULL.
#' @param mu AnvlArray \code{[B, 80, T]} encoder output.
#' @param t Host numeric vector, length B: flow-time in \code{[0, 1]}.
#' @param spks AnvlArray \code{[B, 80]} projected speaker embedding.
#' @param cond AnvlArray \code{[B, 80, T]} prompt-mel conditioning.
#' @param w Weights from \code{\link{yq_cfm_load_weights}}.
#'
#' @return AnvlArray \code{[B, 80, T]} vector-field estimate.
#'
#' @export
yq_cfm_estimator <- function(x, mask, mu, t, spks, cond, w) {
  s <- anvl::shape(x)
  batch <- s[1L]
  seq_len <- s[3L]
  t_emb <- .yq_cfm_time_emb(t, w)

  n_spk <- anvl::shape(spks)[2L]
  spks_exp <- anvl::nv_broadcast_to(
    anvl::nv_reshape(spks, c(batch, n_spk, 1L)), c(batch, n_spk, seq_len))
  h <- anvl::nv_concatenate(x, mu, spks_exp, cond, dimension = 2L)

  # Down stage.
  h <- .yq_cfm_resnet(h, mask, t_emb, w$down_resnet)
  h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  for (blk in w$down_tfm) {
    h <- .yq_cfm_tfm_block(h, blk, batch)
  }
  h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  skip <- h
  h <- .yq_cfm_causal_conv(.yq_cfm_mask_mul(h, mask), w$down_conv_w,
    w$down_conv_b)

  # Mid stages.
  for (m in w$mid) {
    h <- .yq_cfm_resnet(h, mask, t_emb, m$resnet)
    h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
    for (blk in m$tfm) {
      h <- .yq_cfm_tfm_block(h, blk, batch)
    }
    h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  }

  # Up stage (reference order: skip concat -> resnet -> transformers -> conv).
  h <- anvl::nv_concatenate(h, skip, dimension = 2L)
  h <- .yq_cfm_resnet(h, mask, t_emb, w$up_resnet)
  h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  for (blk in w$up_tfm) {
    h <- .yq_cfm_tfm_block(h, blk, batch)
  }
  h <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  h <- .yq_cfm_causal_conv(.yq_cfm_mask_mul(h, mask), w$up_conv_w,
    w$up_conv_b)

  h <- .yq_cfm_block1d(h, mask, w$final_block)
  yunque::conv1d(.yq_cfm_mask_mul(h, mask), w$final_proj_w, w$final_proj_b)
}

# Cosine time grid 1 - cos(linspace(0, 1) * pi/2), f32-rounded at each step
# like the reference's torch f32 scalar chain.
.yq_cfm_t_span <- function(n_timesteps) {
  ts <- .yq_f32(seq(0, 1, length.out = n_timesteps + 1L))
  ang <- .yq_f32(.yq_f32(ts * 0.5) * .yq_f32(pi))
  .yq_f32(1 - .yq_f32(cos(ang)))
}

#' CFM Euler solver with classifier-free guidance (anvl)
#'
#' Torch-free port of \code{CausalConditionalCFM$solve_euler}: fixed-grid
#' Euler integration of the estimator's vector field from t = 0 to 1 on the
#' cosine schedule, each step one batched estimator call (rows 1:B
#' conditional, rows B+1:2B unconditional with zeroed mu/spks/cond) combined
#' as \code{(1 + cfg) * cond - cfg * uncond}. Timestep arithmetic accumulates
#' f32-rounded, matching the reference's torch f32 scalars. Full-length
#' (unpadded) sequences only; the initial noise is an explicit argument
#' (the reference slices a pre-generated noise buffer).
#'
#' @param z AnvlArray \code{[B, 80, T]} initial noise.
#' @param mu AnvlArray \code{[B, 80, T]} encoder output.
#' @param spks AnvlArray \code{[B, 80]} projected speaker embedding.
#' @param cond AnvlArray \code{[B, 80, T]} prompt-mel conditioning.
#' @param w Estimator weights from \code{\link{yq_cfm_load_weights}}.
#' @param n_timesteps Euler steps (reference default 10).
#' @param cfg_rate Classifier-free guidance rate (reference 0.7).
#' @param t_span Optional host numeric time grid, length
#'   \code{n_timesteps + 1} (default: the cosine schedule).
#'
#' @return AnvlArray \code{[B, 80, T]} mel spectrogram.
#'
#' @export
yq_cfm_solve_euler <- function(z, mu, spks, cond, w, n_timesteps = 10L,
                               cfg_rate = 0.7, t_span = NULL) {
  if (is.null(t_span)) {
    t_span <- .yq_cfm_t_span(n_timesteps)
  }
  s <- anvl::shape(mu)
  batch <- s[1L]
  n_mel <- s[2L]
  seq_len <- s[3L]

  # CFG conditioning is fixed across steps: build the 2B-row batch once.
  zeros_bct <- anvl::nv_array(array(0, c(batch, n_mel, seq_len)),
    dtype = "f32")
  zeros_bc <- anvl::nv_array(matrix(0, batch, anvl::shape(spks)[2L]),
    dtype = "f32")
  mu_in <- anvl::nv_concatenate(mu, zeros_bct, dimension = 1L)
  spks_in <- anvl::nv_concatenate(spks, zeros_bc, dimension = 1L)
  cond_in <- anvl::nv_concatenate(cond, zeros_bct, dimension = 1L)

  x <- z
  t <- t_span[1L]
  dt <- .yq_f32(t_span[2L] - t_span[1L])
  n_span <- length(t_span)
  for (step in 2:n_span) {
    x_in <- anvl::nv_concatenate(x, x, dimension = 1L)
    dphi <- yq_cfm_estimator(x_in, NULL, mu_in, rep(t, 2L * batch), spks_in,
      cond_in, w)
    dcond <- anvl::nv_static_slice(dphi, c(1L, 1L, 1L),
      c(batch, n_mel, seq_len), c(1L, 1L, 1L))
    duncond <- anvl::nv_static_slice(dphi, c(batch + 1L, 1L, 1L),
      c(2L * batch, n_mel, seq_len), c(1L, 1L, 1L))
    dphi <- dcond * (1 + cfg_rate) - duncond * cfg_rate
    x <- x + dphi * dt
    t <- .yq_f32(t + dt)
    if (step < n_span) {
      dt <- .yq_f32(t_span[step + 1L] - t_span[step])
    }
  }
  x
}
