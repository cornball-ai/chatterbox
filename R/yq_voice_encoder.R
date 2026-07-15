# anvl/yunque port of the Chatterbox voice encoder (3-layer LSTM speaker
# embedding). Torch-free; yq_ marks the anvl/XLA implementation.

#' Load voice-encoder weights (anvl)
#'
#' @param path Path to ve.safetensors.
#' @return List: 3 LSTM layers (yunque::lstm layout) + projection.
#' @export
yq_ve_load_weights <- function(path) {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, k, transpose = transpose), dtype = "f32")
  }
  layers <- lapply(0:2, function(i) list(
    w_ih = nv(sprintf("lstm.weight_ih_l%d", i), transpose = TRUE),
    w_hh = nv(sprintf("lstm.weight_hh_l%d", i), transpose = TRUE),
    b_ih = nv(sprintf("lstm.bias_ih_l%d", i)),
    b_hh = nv(sprintf("lstm.bias_hh_l%d", i))
  ))
  list(layers = layers, proj_w = nv("proj.weight", transpose = TRUE),
    proj_b = nv("proj.bias"))
}

#' Voice encoder forward (anvl)
#'
#' Torch-free port of the Chatterbox speaker encoder: 3-layer LSTM over mel
#' partials, take the top layer's final hidden state, project, optional
#' ReLU, L2-normalize.
#'
#' @param mels AnvlArray \code{[B, T, num_mels]} (batch-first).
#' @param w Weights from \code{\link{yq_ve_load_weights}}.
#' @param final_relu Logical (chatterbox default TRUE).
#'
#' @return AnvlArray \code{[B, speaker_embed_size]}, L2-normalized.
#'
#' @export
yq_voice_encoder <- function(mels, w, final_relu = TRUE) {
  res <- yunque::lstm(mels, w$layers, batch_first = TRUE)
  hn <- res$h_n # [num_layers, B, hidden]
  s <- anvl::shape(hn)
  batch <- s[2L]
  hidden <- s[3L]
  final <- anvl::nv_reshape(
    anvl::nv_static_slice(hn, start_indices = c(3L, 1L, 1L),
      limit_indices = c(3L, batch, hidden), strides = c(1L, 1L, 1L)),
    c(batch, hidden))
  emb <- yunque::linear(final, w$proj_w, w$proj_b)
  if (final_relu) {
    emb <- anvl::nv_max(emb, 0)
  }
  nrm <- anvl::nv_sqrt(anvl::nv_reduce_sum(emb * emb, dims = 2L, drop = FALSE))
  emb / anvl::nv_broadcast_to(nrm, anvl::shape(emb))
}
