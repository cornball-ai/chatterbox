# anvl/yunque port of the S3Gen flow wrapper (CausalMaskedDiffWithXvec):
# speech-token ids + speaker xvector -> mel spectrogram, via token embedding,
# the (already ported) upsample conformer encoder, encoder projection,
# prompt-mel conditioning, and the CFM Euler/CFG solver in yq_cfm.R.
# Torch-free; batch-1, full-length (unpadded, finalize = TRUE) path only.

#' Load S3Gen flow weights (anvl)
#'
#' Loads everything \code{\link{yq_flow_inference}} needs from
#' s3gen.safetensors: the token-embedding table (kept host-side as an R
#' matrix), speaker affine + encoder projection, the conformer encoder
#' weights (\code{\link{yq_conformer_load_weights}}), and the CFM estimator
#' weights (\code{\link{yq_cfm_load_weights}}).
#'
#' @param path Path to s3gen.safetensors.
#' @param prefix Key prefix (default \code{"flow."}).
#' @return List of weights for \code{\link{yq_flow_inference}}.
#' @export
yq_flow_load_weights <- function(path, prefix = "flow.") {
  st <- yunque::st_open(path)
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  input_embedding <- yunque::st_read(st, paste0(prefix, "input_embedding.weight"))
  spk_w <- nv("spk_embed_affine_layer.weight", TRUE)
  spk_b <- nv("spk_embed_affine_layer.bias")
  proj_w <- nv("encoder_proj.weight", TRUE)
  proj_b <- nv("encoder_proj.bias")
  yunque::st_close(st)
  list(
    input_embedding = input_embedding,
    spk_w = spk_w,
    spk_b = spk_b,
    proj_w = proj_w,
    proj_b = proj_b,
    encoder = yq_conformer_load_weights(path, paste0(prefix, "encoder.")),
    cfm = yq_cfm_load_weights(path, paste0(prefix, "decoder.estimator."))
  )
}

#' S3Gen flow inference: speech tokens -> mel spectrogram (anvl)
#'
#' Torch-free port of \code{CausalMaskedDiffWithXvec$forward} on the batch-1,
#' full-length, \code{finalize = TRUE} path: concatenate prompt and speech
#' tokens, embed (host-side gather), run the upsample conformer encoder (2x
#' in time), project to 80 mel bins, condition the first \code{mel_len1}
#' frames on the prompt mel, and integrate the CFM solver
#' (\code{\link{yq_cfm_solve_euler}}) from the supplied noise. The reference
#' draws its initial noise from a pre-generated buffer; here it is an
#' explicit argument so callers control the randomness.
#'
#' @param speech_tokens Integer vector of speech-token ids (0-based, as
#'   produced by the tokenizer / T3).
#' @param prompt_tokens Integer vector of reference-prompt token ids
#'   (0-based; \code{ref_dict$prompt_token}).
#' @param prompt_feat Prompt mel, \code{[T_mel1, 80]} matrix or
#'   \code{[1, T_mel1, 80]} array (\code{ref_dict$prompt_feat}); T_mel1 is
#'   normally \code{2 * length(prompt_tokens)}.
#' @param embedding Numeric length-192 speaker xvector
#'   (\code{ref_dict$embedding}); L2-normalized internally.
#' @param w Weights from \code{\link{yq_flow_load_weights}}.
#' @param noise Initial CFM noise: \code{[80, T]} matrix or
#'   \code{[1, 80, T]} array with \code{T >= 2 * (n prompt + n speech
#'   tokens)}; extra frames are sliced off (the reference slices its
#'   \code{rand_noise} buffer the same way).
#' @param n_timesteps Euler steps (reference default 10).
#' @param temperature Noise scale (reference default 1.0).
#' @param t_span Optional time grid override for
#'   \code{\link{yq_cfm_solve_euler}}.
#' @param vocab_size Token vocabulary (default 6561); ids clamp to range.
#' @param token_mel_ratio Mel frames per token (default 2).
#'
#' @return AnvlArray \code{[1, 80, 2 * length(speech_tokens)]}: the generated
#'   mel (prompt frames already stripped).
#'
#' @export
yq_flow_inference <- function(speech_tokens, prompt_tokens, prompt_feat,
                              embedding, w, noise, n_timesteps = 10L,
                              temperature = 1.0, t_span = NULL,
                              vocab_size = 6561L, token_mel_ratio = 2L) {
  tokens <- c(as.integer(prompt_tokens), as.integer(speech_tokens))
  tokens <- pmin(pmax(tokens, 0L), vocab_size - 1L)
  n_tok <- length(tokens)

  # Speaker path: L2-normalize the xvector (host), then affine to 80 dims.
  e <- as.numeric(embedding)
  e <- e / max(sqrt(sum(e^2)), 1e-12)
  spks <- yunque::linear(anvl::nv_array(matrix(e, 1L), dtype = "f32"),
    w$spk_w, w$spk_b) # [1, 80]

  # Token embedding: host-side gather (+1: R matrices are 1-indexed).
  emb <- w$input_embedding[tokens + 1L, , drop = FALSE]
  x <- anvl::nv_array(array(emb, c(1L, n_tok, ncol(emb))), dtype = "f32")

  # Encoder (2x upsampling in time) + projection to mel bins.
  h <- yq_conformer(x, w$encoder) # [1, 2T, 512]
  h <- yunque::linear(h, w$proj_w, w$proj_b) # [1, 2T, 80]

  # Prompt conditioning spans the actual prompt mel length.
  pf <- prompt_feat
  if (length(dim(pf)) == 3L) {
    pf <- array(pf[1L, , ], dim(pf)[2:3])
  }
  mel_len1 <- nrow(pf)
  mel_total <- token_mel_ratio * n_tok
  mel_len2 <- mel_total - mel_len1
  cond_arr <- array(0, c(1L, 80L, mel_total))
  if (mel_len1 > 0L) {
    cond_arr[1L, , seq_len(mel_len1)] <- t(pf)
  }
  cond <- anvl::nv_array(cond_arr, dtype = "f32")

  mu <- anvl::nv_transpose(h, c(1L, 3L, 2L)) # [1, 80, mel_total]

  # Initial noise: slice the leading mel_total frames, like the reference's
  # rand_noise buffer slice.
  zn <- noise
  if (length(dim(zn)) == 3L) {
    zn <- array(zn[1L, , ], dim(zn)[2:3])
  }
  z_arr <- array(zn[, seq_len(mel_total), drop = FALSE],
    c(1L, 80L, mel_total))
  z <- anvl::nv_array(z_arr, dtype = "f32")
  if (temperature != 1.0) {
    z <- z * temperature
  }

  feat <- yq_cfm_solve_euler(z, mu, spks, cond, w$cfm,
    n_timesteps = n_timesteps, t_span = t_span)

  # Generated portion (after the prompt frames).
  anvl::nv_static_slice(feat, c(1L, 1L, mel_len1 + 1L),
    c(1L, 80L, mel_len1 + mel_len2), c(1L, 1L, 1L))
}
