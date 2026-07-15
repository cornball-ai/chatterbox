# anvl/yunque port of the T3 conditioning encoder (perceiver resampler +
# speaker/emotion). Composed from yunque primitives -- the recurring MHA
# block is tracked for promotion in yunque#8, glue for now.

# Cross/self attention block: shared affine LayerNorm on both inputs, q<-x1,
# k/v<-x2, out-proj, residual on x1.
.yq_attn_block <- function(x1, x2, a, n_head) {
  eps <- 1e-5
  s1 <- anvl::shape(x1)
  batch <- s1[1L]
  q_len <- s1[2L]
  dim <- s1[3L]
  hd <- dim %/% n_head
  x1n <- yunque::layer_norm(x1, a$norm_w, a$norm_b, eps)
  x2n <- yunque::layer_norm(x2, a$norm_w, a$norm_b, eps)
  q <- .yq_heads(yunque::linear(x1n, a$qw, a$qb), batch, n_head, hd)
  k <- .yq_heads(yunque::linear(x2n, a$kw, a$kb), batch, n_head, hd)
  v <- .yq_heads(yunque::linear(x2n, a$vw, a$vb), batch, n_head, hd)
  o <- yunque::sdpa(q, k, v)
  o <- anvl::nv_reshape(anvl::nv_transpose(o, c(1L, 3L, 2L, 4L)),
    c(batch, q_len, dim))
  x1 + yunque::linear(o, a$ow, a$ob)
}

# Perceiver: learned queries cross-attend to x, then self-attend (shared block).
.yq_perceiver <- function(x, query, a, n_head = 4L) {
  qs <- anvl::shape(query)
  q <- anvl::nv_broadcast_to(query, c(anvl::shape(x)[1L], qs[2L], qs[3L]))
  pre <- .yq_attn_block(q, x, a, n_head)
  .yq_attn_block(pre, pre, a, n_head)
}

#' Load T3 conditioning-encoder weights (anvl)
#' @param path Path to t3_cfg.safetensors.
#' @param prefix Key prefix (default \code{"cond_enc."}).
#' @export
yq_t3_cond_load_weights <- function(path, prefix = "cond_enc.") {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  attn <- list(
    norm_w = nv("perceiver.attn.norm.weight"),
    norm_b = nv("perceiver.attn.norm.bias"),
    qw = nv("perceiver.attn.to_q.weight", TRUE),
    qb = nv("perceiver.attn.to_q.bias"),
    kw = nv("perceiver.attn.to_k.weight", TRUE),
    kb = nv("perceiver.attn.to_k.bias"),
    vw = nv("perceiver.attn.to_v.weight", TRUE),
    vb = nv("perceiver.attn.to_v.bias"),
    ow = nv("perceiver.attn.proj_out.weight", TRUE),
    ob = nv("perceiver.attn.proj_out.bias")
  )
  list(
    spkr_w = nv("spkr_enc.weight", TRUE), spkr_b = nv("spkr_enc.bias"),
    emotion_w = nv("emotion_adv_fc.weight", TRUE),
    query = nv("perceiver.pre_attention_query"),
    attn = attn
  )
}

#' T3 conditioning encoder forward (anvl)
#'
#' Torch-free port of \code{t3_cond_enc}: speaker projection (1 token) +
#' perceiver-resampled prompt speech (32 tokens) + emotion (1 token),
#' concatenated to \code{[B, n_cond, 1024]}.
#'
#' @param speaker_emb AnvlArray \code{[B, speaker_embed_size]}.
#' @param cond_prompt_speech_emb AnvlArray \code{[B, T, 1024]} or NULL.
#' @param emotion Numeric emotion/exaggeration control.
#' @param w Weights from \code{\link{yq_t3_cond_load_weights}}.
#'
#' @return AnvlArray \code{[B, n_cond, 1024]}.
#'
#' @export
yq_t3_cond_enc <- function(speaker_emb, cond_prompt_speech_emb, emotion, w) {
  spkr <- anvl::nv_unsqueeze(yunque::linear(speaker_emb, w$spkr_w, w$spkr_b), 2L)
  parts <- list(spkr)
  if (!is.null(cond_prompt_speech_emb)) {
    parts <- c(parts, list(.yq_perceiver(cond_prompt_speech_emb, w$query, w$attn)))
  }
  em <- yunque::linear(
    anvl::nv_array(array(as.numeric(emotion), c(1L, 1L, 1L)), dtype = "f32"),
    w$emotion_w)
  parts <- c(parts, list(em))
  do.call(anvl::nv_concatenate, c(parts, list(dimension = 2L)))
}
