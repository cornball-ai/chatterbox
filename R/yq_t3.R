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

# embed tokens [B,L] + learned positions -> [B, L, dim]. zero_tok zeros the
# token embedding (CFG unconditioned branch keeps only the position emb).
.yq_t3_embed <- function(emb, pos, tokens, zero_tok = FALSE) {
  tokens <- matrix(as.integer(tokens), nrow = nrow(tokens))
  n <- ncol(tokens)
  x <- yunque::embedding(emb, tokens)
  if (zero_tok) {
    x <- x * 0
  }
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
yq_t3_forward <- function(cond_emb, text_tokens, speech_tokens, w, llama_w,
                          zero_text = FALSE) {
  te <- .yq_t3_embed(w$text_emb, w$text_pos, text_tokens, zero_tok = zero_text)
  se <- .yq_t3_embed(w$speech_emb, w$speech_pos, speech_tokens)
  embeds <- anvl::nv_concatenate(cond_emb, te, se, dimension = 2L)
  hidden <- yq_llama(embeds, llama_w)
  lc <- anvl::shape(cond_emb)[2L]
  lt <- anvl::shape(te)[2L]
  ls <- anvl::shape(se)[2L]
  speech_latents <- yunque::slice_seq(hidden, lc + lt + 1L, lc + lt + ls)
  yunque::linear(speech_latents, w$speech_head)
}

# Base-R port of .sample_speech_token: sign-dependent repetition penalty ->
# temperature -> min-p -> top-p -> multinomial. logits/generated 0-based.
.yq_sample_speech_token <- function(logits, generated, temperature, top_p,
                                    min_p, repetition_penalty) {
  if (repetition_penalty != 1 && length(generated) > 0L) {
    ids <- unique(generated) + 1L
    v <- logits[ids]
    logits[ids] <- ifelse(v > 0, v / repetition_penalty, v * repetition_penalty)
  }
  if (temperature != 1) {
    logits <- logits / temperature
  }
  probs <- exp(logits - max(logits))
  probs <- probs / sum(probs)
  probs[probs < min_p * max(probs)] <- 0
  probs <- probs / sum(probs)
  ord <- order(probs, decreasing = TRUE)
  sp <- probs[ord]
  if (top_p < 1) {
    cs <- cumsum(sp)
    mask <- c(FALSE, utils::head(cs > top_p, -1L)) # HF shifts the mask right 1
    sp[mask] <- 0
    sp <- sp / sum(sp)
  }
  ord[sample.int(length(sp), 1L, prob = sp)] - 1L
}

#' Generate speech tokens (anvl, CFG + sampling)
#'
#' Autoregressive T3 generation: recompute the forward over the growing
#' speech sequence (O(S^2); the KV cache is a later optimization), combine
#' conditioned/unconditioned logits by classifier-free guidance, and sample
#' via \code{.yq_sample_speech_token}. Faithful to \code{t3_inference}.
#'
#' @param cond_emb AnvlArray \code{[1, len_cond, 1024]}.
#' @param text_tokens Integer matrix \code{[1, len_text]} (0-based, already
#'   padded with start/stop text tokens by the caller).
#' @param w,llama_w Weights from \code{\link{yq_t3_load_weights}} /
#'   \code{\link{yq_llama_load_weights}}.
#' @param config T3 config (for start/stop speech tokens).
#' @param max_new,temperature,cfg_weight,top_p,min_p,repetition_penalty
#'   Sampling controls (defaults match \code{t3_inference}).
#'
#' @return Integer vector of generated 0-based speech tokens.
#'
#' @export
yq_t3_generate <- function(cond_emb, text_tokens, w, llama_w, config,
                           max_new = 200L, temperature = 0.8, cfg_weight = 0.5,
                           top_p = 1, min_p = 0.05, repetition_penalty = 1.2) {
  generated <- integer(0)
  speech <- matrix(config$start_speech_token, nrow = 1L)
  for (step in seq_len(max_new)) {
    ls <- ncol(speech)
    lc <- as.array(yq_t3_forward(cond_emb, text_tokens, speech, w, llama_w))[1L, ls, ]
    if (cfg_weight > 0) {
      lu <- as.array(yq_t3_forward(cond_emb, text_tokens, speech, w, llama_w,
        zero_text = TRUE))[1L, ls, ]
      lc <- lu + cfg_weight * (lc - lu)
    }
    nxt <- .yq_sample_speech_token(lc, generated, temperature, top_p, min_p,
      repetition_penalty)
    if (nxt == config$stop_speech_token) {
      break
    }
    generated <- c(generated, nxt)
    speech <- cbind(speech, nxt)
  }
  generated
}
