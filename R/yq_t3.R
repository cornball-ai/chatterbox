# anvl/yunque port of the T3 wrapper forward: text/speech embeddings +
# learned positions + assembly [cond | text | speech] -> Llama backbone ->
# speech head. Generation loop (CFG + sampling) below.

#' Load T3 wrapper weights (anvl)
#'
#' Embedding + position tables stay R matrices for host-side gather; the
#' output heads are anvl \code{[in, out]} for \code{yunque::linear}.
#'
#' @param path Path to t3_cfg.safetensors.
#' @return List of embeddings, position tables, and heads.
#' @export
yq_t3_load_weights <- function(path) {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  list(
    text_emb = yunque::st_read(st, "text_emb.weight"),
    speech_emb = yunque::st_read(st, "speech_emb.weight"),
    text_pos = yunque::st_read(st, "text_pos_emb.emb.weight"),
    speech_pos = yunque::st_read(st, "speech_pos_emb.emb.weight"),
    text_head = anvl::nv_array(
      yunque::st_read(st, "text_head.weight", transpose = TRUE), dtype = "f32"),
    speech_head = anvl::nv_array(
      yunque::st_read(st, "speech_head.weight", transpose = TRUE), dtype = "f32")
  )
}

# embed tokens [B,L] + learned positions -> [B, L, dim]
.yq_t3_embed <- function(emb, pos, tokens) {
  tokens <- matrix(as.integer(tokens), nrow = nrow(tokens))
  n <- ncol(tokens)
  x <- yunque::embedding(emb, tokens)
  p <- anvl::nv_unsqueeze(yunque::embedding(pos, seq_len(n) - 1L), 1L)
  x + anvl::nv_broadcast_to(p, anvl::shape(x))
}

#' T3 forward: speech logits (anvl)
#'
#' Torch-free port of \code{t3_model$forward}: assemble
#' \code{[cond | text_emb+pos | speech_emb+pos]}, run the Llama backbone,
#' project the speech positions through the speech head.
#'
#' @param cond_emb AnvlArray \code{[B, len_cond, 1024]} (from
#'   \code{\link{yq_t3_cond_enc}}).
#' @param text_tokens Integer matrix \code{[B, len_text]} (0-based).
#' @param speech_tokens Integer matrix \code{[B, len_speech]} (0-based).
#' @param w Weights from \code{\link{yq_t3_load_weights}}.
#' @param llama_w Weights from \code{\link{yq_llama_load_weights}}.
#'
#' @return AnvlArray speech logits \code{[B, len_speech, speech_vocab]}.
#'
#' @export
yq_t3_forward <- function(cond_emb, text_tokens, speech_tokens, w, llama_w) {
  te <- .yq_t3_embed(w$text_emb, w$text_pos, text_tokens)
  se <- .yq_t3_embed(w$speech_emb, w$speech_pos, speech_tokens)
  embeds <- anvl::nv_concatenate(cond_emb, te, se, dimension = 2L)
  hidden <- yq_llama(embeds, llama_w)
  lc <- anvl::shape(cond_emb)[2L]
  lt <- anvl::shape(te)[2L]
  ls <- anvl::shape(se)[2L]
  speech_latents <- yunque::slice_seq(hidden, lc + lt + 1L, lc + lt + ls)
  yunque::linear(speech_latents, w$speech_head)
}
