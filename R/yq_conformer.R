# anvl/yunque port of the S3Gen UpsampleConformerEncoder (flow.encoder):
# LinearNoSubsampling -> pre-lookahead -> 6 conformer blocks -> Upsample1D 2x
# -> up_embed -> 4 conformer blocks -> after_norm. ESPnet relative-position
# attention (yunque::rel_position_attention). Torch-free; yq_ marks the port.
# Batch-1, full-length (unpadded) path -- padding masks are not applied.

# [B, S, D] -> [B, H, S, hd]
.yq_heads <- function(x, batch, n_head, head_dim) {
  s <- anvl::shape(x)
  anvl::nv_transpose(anvl::nv_reshape(x, c(batch, s[2L], n_head, head_dim)),
    c(1L, 3L, 2L, 4L))
}

# Leaky ReLU (torch default negative slope 0.01); no yunque primitive.
.yq_leaky_relu <- function(x, slope = 0.01) {
  anvl::nv_max(x, 0) + anvl::nv_min(x, 0) * slope
}

# 1-D nearest upsample by an integer factor over [B, C, W].
.yq_upsample1d_nearest <- function(x, factor = 2L) {
  s <- anvl::shape(x)
  b <- s[1L]; c <- s[2L]; w <- s[3L]
  x <- anvl::nv_reshape(x, c(b, c, w, 1L))
  x <- anvl::nv_broadcast_to(x, c(b, c, w, factor))
  anvl::nv_reshape(x, c(b, c, w * factor))
}

# ESPnet relative positional-encoding table for a length-T sequence.
# Returns an R matrix [2T-1, d_model]; relative positions run
# +(T-1), ..., 0, ..., -(T-1) down the rows (matches the buffer slice the
# torch encoder takes from its precomputed pe).
.yq_conformer_pos_emb <- function(seq_len, d_model = 512L) {
  idx <- seq(0, d_model - 2L, by = 2L)                 # 0,2,...,d-2 (len d/2)
  div_term <- exp(idx * (-(log(10000) / d_model)))     # len d/2
  rel <- (seq_len - 1L):(-(seq_len - 1L))              # len 2T-1
  ang <- outer(rel, div_term)                          # [2T-1, d/2]
  pe <- matrix(0, length(rel), d_model)
  pe[, seq(1L, d_model, by = 2L)] <- sin(ang)
  pe[, seq(2L, d_model, by = 2L)] <- cos(ang)
  pe
}

# LinearNoSubsampling: Linear -> LayerNorm(eps 1e-5) -> scale by sqrt(d_model).
.yq_lns <- function(x, e, d_model = 512L) {
  h <- yunque::linear(x, e$lin_w, e$lin_b)
  h <- yunque::layer_norm(h, e$ln_w, e$ln_b, eps = 1e-5)
  h * sqrt(d_model)
}

# One conformer block (attention + FFN, both pre-norm LayerNorm eps 1e-12).
# pos_emb is the [1, 2T-1, D] positional table (anvl); each block projects it
# through its own linear_pos.
.yq_conf_block <- function(x, pos_emb, blk, n_head, batch, seq_len) {
  s <- anvl::shape(x)
  d <- s[3L]
  hd <- d %/% n_head

  # relative-position keys p: [1, H, 2T-1, hd]
  p <- yunque::linear(pos_emb, blk$pos_w)
  p <- .yq_heads(p, 1L, n_head, hd)

  res <- x
  xn <- yunque::layer_norm(x, blk$norm_mha_w, blk$norm_mha_b, eps = 1e-12)
  q <- .yq_heads(yunque::linear(xn, blk$q_w, blk$q_b), batch, n_head, hd)
  k <- .yq_heads(yunque::linear(xn, blk$k_w, blk$k_b), batch, n_head, hd)
  v <- .yq_heads(yunque::linear(xn, blk$v_w, blk$v_b), batch, n_head, hd)
  a <- yunque::rel_position_attention(q, k, v, p, blk$pos_bias_u, blk$pos_bias_v)
  a <- anvl::nv_reshape(anvl::nv_transpose(a, c(1L, 3L, 2L, 4L)),
    c(batch, seq_len, d))
  a <- yunque::linear(a, blk$out_w, blk$out_b)
  x <- res + a

  res <- x
  xn <- yunque::layer_norm(x, blk$norm_ff_w, blk$norm_ff_b, eps = 1e-12)
  ff <- yunque::linear(yunque::silu(yunque::linear(xn, blk$w1_w, blk$w1_b)),
    blk$w2_w, blk$w2_b)
  res + ff
}

#' Load UpsampleConformerEncoder weights (anvl)
#'
#' @param path Path to s3gen.safetensors.
#' @param prefix Key prefix (default \code{"flow.encoder."}).
#' @param num_blocks Conformer blocks before upsample (default 6).
#' @param num_up_blocks Conformer blocks after upsample (default 4).
#' @return List of weights for \code{\link{yq_conformer}}.
#' @export
yq_conformer_load_weights <- function(path, prefix = "flow.encoder.",
                                      num_blocks = 6L, num_up_blocks = 4L) {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  embed <- function(pre) list(
    lin_w = nv(paste0(pre, "out.0.weight"), TRUE),
    lin_b = nv(paste0(pre, "out.0.bias")),
    ln_w = nv(paste0(pre, "out.1.weight")),
    ln_b = nv(paste0(pre, "out.1.bias"))
  )
  block <- function(pre) list(
    q_w = nv(paste0(pre, "self_attn.linear_q.weight"), TRUE),
    q_b = nv(paste0(pre, "self_attn.linear_q.bias")),
    k_w = nv(paste0(pre, "self_attn.linear_k.weight"), TRUE),
    k_b = nv(paste0(pre, "self_attn.linear_k.bias")),
    v_w = nv(paste0(pre, "self_attn.linear_v.weight"), TRUE),
    v_b = nv(paste0(pre, "self_attn.linear_v.bias")),
    out_w = nv(paste0(pre, "self_attn.linear_out.weight"), TRUE),
    out_b = nv(paste0(pre, "self_attn.linear_out.bias")),
    pos_w = nv(paste0(pre, "self_attn.linear_pos.weight"), TRUE),
    pos_bias_u = nv(paste0(pre, "self_attn.pos_bias_u")),
    pos_bias_v = nv(paste0(pre, "self_attn.pos_bias_v")),
    w1_w = nv(paste0(pre, "feed_forward.w_1.weight"), TRUE),
    w1_b = nv(paste0(pre, "feed_forward.w_1.bias")),
    w2_w = nv(paste0(pre, "feed_forward.w_2.weight"), TRUE),
    w2_b = nv(paste0(pre, "feed_forward.w_2.bias")),
    norm_mha_w = nv(paste0(pre, "norm_mha.weight")),
    norm_mha_b = nv(paste0(pre, "norm_mha.bias")),
    norm_ff_w = nv(paste0(pre, "norm_ff.weight")),
    norm_ff_b = nv(paste0(pre, "norm_ff.bias"))
  )
  list(
    embed = embed("embed."),
    pre_conv1_w = nv("pre_lookahead_layer.conv1.weight"),
    pre_conv1_b = nv("pre_lookahead_layer.conv1.bias"),
    pre_conv2_w = nv("pre_lookahead_layer.conv2.weight"),
    pre_conv2_b = nv("pre_lookahead_layer.conv2.bias"),
    encoders = lapply(seq_len(num_blocks) - 1L,
      function(i) block(sprintf("encoders.%d.", i))),
    up_conv_w = nv("up_layer.conv.weight"),
    up_conv_b = nv("up_layer.conv.bias"),
    up_embed = embed("up_embed."),
    up_encoders = lapply(seq_len(num_up_blocks) - 1L,
      function(i) block(sprintf("up_encoders.%d.", i))),
    after_norm_w = nv("after_norm.weight"),
    after_norm_b = nv("after_norm.bias")
  )
}

#' UpsampleConformerEncoder forward (anvl)
#'
#' Torch-free port of the S3Gen \code{UpsampleConformerEncoder}: input
#' embedding + ESPnet relative positional encoding, causal pre-lookahead
#' convolutions, 6 conformer blocks, 2x nearest upsample + conv, a second
#' embedding, 4 more conformer blocks, final LayerNorm. Batch-1, full-length
#' (unpadded) only -- no attention/padding masks.
#'
#' @param x AnvlArray \code{[B, T, D]} (speech-token embeddings + xvector,
#'   summed upstream). D = 512.
#' @param w Weights from \code{\link{yq_conformer_load_weights}}.
#' @param n_head Attention heads (default 8).
#' @param pre_lookahead_len Look-ahead length (default 3).
#'
#' @return AnvlArray \code{[B, 2T, D]}.
#'
#' @export
yq_conformer <- function(x, w, n_head = 8L, pre_lookahead_len = 3L) {
  s <- anvl::shape(x)
  batch <- s[1L]
  seq_len <- s[2L]
  d <- s[3L]

  # Input embedding + positional table.
  xs <- .yq_lns(x, w$embed, d)
  pos_emb <- anvl::nv_array(
    array(.yq_conformer_pos_emb(seq_len, d), c(1L, 2L * seq_len - 1L, d)),
    dtype = "f32")

  # Pre-lookahead: [B,T,D] -> [B,D,T], right-pad + conv1 (k=4) leaky-relu,
  # left-pad + conv2 (k=3), residual.
  h <- anvl::nv_transpose(xs, c(1L, 3L, 2L))
  h <- anvl::nv_pad(h, 0, c(0L, 0L, 0L), c(0L, 0L, pre_lookahead_len))
  h <- .yq_leaky_relu(yunque::conv1d(h, w$pre_conv1_w, w$pre_conv1_b))
  h <- anvl::nv_pad(h, 0, c(0L, 0L, 2L), c(0L, 0L, 0L))
  h <- yunque::conv1d(h, w$pre_conv2_w, w$pre_conv2_b)
  xs <- xs + anvl::nv_transpose(h, c(1L, 3L, 2L))

  # First conformer stack.
  for (blk in w$encoders) {
    xs <- .yq_conf_block(xs, pos_emb, blk, n_head, batch, seq_len)
  }

  # Upsample 2x: [B,T,D] -> [B,D,T], nearest x2, left-pad 4, conv (k=5).
  h <- anvl::nv_transpose(xs, c(1L, 3L, 2L))
  h <- .yq_upsample1d_nearest(h, 2L)
  h <- anvl::nv_pad(h, 0, c(0L, 0L, 4L), c(0L, 0L, 0L))
  h <- yunque::conv1d(h, w$up_conv_w, w$up_conv_b)
  xs <- anvl::nv_transpose(h, c(1L, 3L, 2L))
  seq_up <- 2L * seq_len

  # Second embedding + positional table.
  xs <- .yq_lns(xs, w$up_embed, d)
  pos_emb_up <- anvl::nv_array(
    array(.yq_conformer_pos_emb(seq_up, d), c(1L, 2L * seq_up - 1L, d)),
    dtype = "f32")

  # Second conformer stack.
  for (blk in w$up_encoders) {
    xs <- .yq_conf_block(xs, pos_emb_up, blk, n_head, batch, seq_up)
  }

  yunque::layer_norm(xs, w$after_norm_w, w$after_norm_b, eps = 1e-5)
}
