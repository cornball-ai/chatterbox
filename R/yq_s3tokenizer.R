# anvl/yunque port of the S3 tokenizer (S3AudioEncoderV2 + FSQ quantizer).
# Two strided convs downsample the log-mel, six FSMN-attention blocks
# (multi-head attn with split-half RoPE + a depthwise temporal-memory
# conv), then FSQ base-3 quantization -> discrete speech token ids.
# Torch-free; yq_ marks the anvl/XLA implementation.

# Half-dim cos/sin tables [seq, head_dim/2] for yunque::rope_split. The torch
# code cats [freqs, freqs] to head_dim; the two halves are identical, so the
# half table is all rope_split needs.
.yq_s3_rope <- function(seq_len, head_dim = 64L, theta = 10000) {
  inv_freq <- 1 / theta^(seq(0, head_dim - 2, by = 2) / head_dim)
  freqs <- outer(0:(seq_len - 1L), inv_freq)
  list(cos = cos(freqs), sin = sin(freqs))
}

#' Load S3 tokenizer weights (anvl)
#'
#' Linear weights are read \code{[out, in] -> [in, out]} for
#' \code{yunque::linear}; conv weights \code{[out, in, k]} stay as-is;
#' norm/bias vectors as-is.
#'
#' @param path Path to s3gen.safetensors.
#' @param prefix Key prefix (default \code{"tokenizer."}).
#' @param n_layer Number of encoder blocks (default 6).
#' @return List of conv, block, and quantizer weights.
#' @export
yq_s3tokenizer_load_weights <- function(path, prefix = "tokenizer.",
                                        n_layer = 6L) {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  blocks <- lapply(seq_len(n_layer) - 1L, function(i) {
    p <- sprintf("encoder.blocks.%d.", i)
    list(
      attn_ln_w = nv(paste0(p, "attn_ln.weight")),
      attn_ln_b = nv(paste0(p, "attn_ln.bias")),
      qw = nv(paste0(p, "attn.query.weight"), TRUE),
      qb = nv(paste0(p, "attn.query.bias")),
      kw = nv(paste0(p, "attn.key.weight"), TRUE),
      vw = nv(paste0(p, "attn.value.weight"), TRUE),
      vb = nv(paste0(p, "attn.value.bias")),
      ow = nv(paste0(p, "attn.out.weight"), TRUE),
      ob = nv(paste0(p, "attn.out.bias")),
      fsmn = nv(paste0(p, "attn.fsmn_block.weight")),
      mlp_ln_w = nv(paste0(p, "mlp_ln.weight")),
      mlp_ln_b = nv(paste0(p, "mlp_ln.bias")),
      mlp0_w = nv(paste0(p, "mlp.0.weight"), TRUE),
      mlp0_b = nv(paste0(p, "mlp.0.bias")),
      mlp2_w = nv(paste0(p, "mlp.2.weight"), TRUE),
      mlp2_b = nv(paste0(p, "mlp.2.bias"))
    )
  })
  list(
    conv1_w = nv("encoder.conv1.weight"),
    conv1_b = nv("encoder.conv1.bias"),
    conv2_w = nv("encoder.conv2.weight"),
    conv2_b = nv("encoder.conv2.bias"),
    blocks = blocks,
    proj_down_w = nv("quantizer._codebook.project_down.weight", TRUE),
    proj_down_b = nv("quantizer._codebook.project_down.bias")
  )
}

# FSMN depthwise temporal-memory conv on the raw value projection
# [B, seq, C]: (B,seq,C) -> (B,C,seq) -> pad+depthwise conv (k=31, sym pad
# 15) -> (B,seq,C) + residual.
.yq_s3_fsmn <- function(v, fsmn_w) {
  s <- anvl::shape(v)
  x <- anvl::nv_transpose(v, c(1L, 3L, 2L)) # [B, C, seq]
  x <- yunque::conv1d(x, fsmn_w, padding = 15L, groups = s[3L])
  x <- anvl::nv_transpose(x, c(1L, 3L, 2L)) # [B, seq, C]
  x + v
}

# One FSMN residual-attention block. attn_ln eps 1e-6 (explicit in torch);
# mlp_ln eps 1e-5 (nn_layer_norm default). MLP gelu is exact erf (nn_gelu
# default approximate = "none").
.yq_s3_block <- function(x, b, cos, sin, n_head) {
  s <- anvl::shape(x)
  batch <- s[1L]
  seq <- s[2L]
  dim <- s[3L]
  hd <- dim %/% n_head

  h <- yunque::layer_norm(x, b$attn_ln_w, b$attn_ln_b, 1e-6)
  q <- yunque::linear(h, b$qw, b$qb)
  k <- yunque::linear(h, b$kw)
  v <- yunque::linear(h, b$vw, b$vb)

  fsm <- .yq_s3_fsmn(v, b$fsmn)

  qh <- yunque::rope_split(.yq_heads(q, batch, n_head, hd), cos, sin)
  kh <- yunque::rope_split(.yq_heads(k, batch, n_head, hd), cos, sin)
  vh <- .yq_heads(v, batch, n_head, hd)
  o <- yunque::sdpa(qh, kh, vh)
  o <- anvl::nv_reshape(anvl::nv_transpose(o, c(1L, 3L, 2L, 4L)),
    c(batch, seq, dim))
  x <- x + (yunque::linear(o, b$ow, b$ob) + fsm)

  h2 <- yunque::layer_norm(x, b$mlp_ln_w, b$mlp_ln_b, 1e-5)
  mlp <- yunque::linear(
    yunque::gelu(yunque::linear(h2, b$mlp0_w, b$mlp0_b), approximate = "none"),
    b$mlp2_w, b$mlp2_b)
  x + mlp
}

#' S3 audio encoder forward (anvl)
#'
#' Torch-free port of \code{s3_audio_encoder$forward} for a single
#' full-length (unpadded) mel: two strided convs (gelu) downsample by 4,
#' then \code{n_head}-head FSMN-attention blocks with split-half RoPE.
#' Padding masks are identity for a full-length sequence and omitted.
#'
#' @param mel AnvlArray \code{[B, n_mels, T]} log-mel.
#' @param w Weights from \code{\link{yq_s3tokenizer_load_weights}}.
#' @param n_head Attention heads (default 20).
#'
#' @return AnvlArray hidden states \code{[B, seq, n_state]}.
#'
#' @export
yq_s3tokenizer_encode <- function(mel, w, n_head = 20L) {
  x <- yunque::gelu(
    yunque::conv1d(mel, w$conv1_w, w$conv1_b, stride = 2L, padding = 1L),
    approximate = "none")
  x <- yunque::gelu(
    yunque::conv1d(x, w$conv2_w, w$conv2_b, stride = 2L, padding = 1L),
    approximate = "none")
  x <- anvl::nv_transpose(x, c(1L, 3L, 2L)) # [B, seq, C]

  s <- anvl::shape(x)
  batch <- s[1L]
  seq <- s[2L]
  hd <- s[3L] %/% n_head
  rope <- .yq_s3_rope(seq, hd)
  bc <- function(m) anvl::nv_broadcast_to(
    anvl::nv_reshape(anvl::nv_array(m, dtype = "f32"), c(1L, 1L, seq, hd %/% 2L)),
    c(batch, n_head, seq, hd %/% 2L))
  cos <- bc(rope$cos)
  sin <- bc(rope$sin)

  for (b in w$blocks) {
    x <- .yq_s3_block(x, b, cos, sin, n_head)
  }
  x
}

# FSQ base-3 quantizer: project_down (1280->8), tanh, round to {0,1,2},
# base-3 dot with 3^(0:7) -> a single 0..6560 code per frame. Round is
# host-side; the arithmetic matches torch exactly.
.yq_s3_fsq <- function(proj) {
  h <- tanh(proj) * 0.9990000128746033
  h <- round(h) + 1
  powers <- 3^(0:7)
  d <- dim(h)
  hm <- matrix(h, ncol = d[length(d)])
  mu <- as.vector(hm %*% powers)
  array(as.integer(round(mu)), dim = d[-length(d)])
}

#' S3 tokenizer forward (anvl)
#'
#' Full torch-free port: encode the log-mel then FSQ-quantize the hidden
#' states to discrete speech token ids (0-based, vocab 6561 = 3^8).
#'
#' @param mel AnvlArray \code{[B, n_mels, T]} log-mel.
#' @param w Weights from \code{\link{yq_s3tokenizer_load_weights}}.
#' @param n_head Attention heads (default 20).
#'
#' @return List: \code{hidden} (AnvlArray \code{[B, seq, n_state]}) and
#'   \code{tokens} (integer array \code{[B, seq]}).
#'
#' @export
yq_s3tokenizer <- function(mel, w, n_head = 20L) {
  hidden <- yq_s3tokenizer_encode(mel, w, n_head)
  proj <- as.array(yunque::linear(hidden, w$proj_down_w, w$proj_down_b))
  list(hidden = hidden, tokens = .yq_s3_fsq(proj))
}
