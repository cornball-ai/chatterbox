# anvl/yunque port of the T3 Llama backbone (520M): RoPE (llama3-scaled) +
# RMSNorm + SwiGLU, plain MHA. Torch-free; yq_ marks the anvl impl.

# [B, S, hidden] -> [B, n_head, S, head_dim]
.yq_heads <- function(x, batch, n_head, head_dim) {
  s <- anvl::shape(x)
  anvl::nv_transpose(anvl::nv_reshape(x, c(batch, s[2L], n_head, head_dim)),
    c(1L, 3L, 2L, 4L))
}

# llama3 RoPE frequency scaling (per-frequency, by wavelength).
.llama3_scale <- function(inv_freq, factor, low_f, high_f, old_ctx) {
  low_wl <- old_ctx / low_f
  high_wl <- old_ctx / high_f
  wavelen <- 2 * pi / inv_freq
  vapply(seq_along(inv_freq), function(i) {
    wl <- wavelen[i]
    if (wl < high_wl) {
      inv_freq[i]
    } else if (wl > low_wl) {
      inv_freq[i] / factor
    } else {
      smooth <- (old_ctx / wl - low_f) / (high_f - low_f)
      (1 - smooth) * inv_freq[i] / factor + smooth * inv_freq[i]
    }
  }, numeric(1))
}

# Half-dim cos/sin tables [seq, head_dim/2] for yunque::rope_split (the torch
# code cats [freqs, freqs] to head_dim; the two halves are identical).
.yq_llama_rope <- function(seq_len, head_dim = 64L, theta = 5e5) {
  inv_freq <- 1 / theta^(seq(0, head_dim - 2, by = 2) / head_dim)
  inv_freq <- .llama3_scale(inv_freq, factor = 8, low_f = 1, high_f = 4,
    old_ctx = 8192)
  freqs <- outer(0:(seq_len - 1L), inv_freq)
  list(cos = cos(freqs), sin = sin(freqs))
}

#' Load T3 Llama backbone weights (anvl)
#'
#' @param path Path to t3_cfg.safetensors.
#' @param prefix Key prefix (default \code{"tfmr."}).
#' @param n_layers Number of decoder layers.
#' @return List of per-layer weights + final norm.
#' @export
yq_llama_load_weights <- function(path, prefix = "tfmr.", n_layers = 30L) {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k, transpose = FALSE) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = transpose),
      dtype = "f32")
  }
  layers <- lapply(seq_len(n_layers) - 1L, function(i) {
    p <- sprintf("layers.%d.", i)
    list(
      input_ln = nv(paste0(p, "input_layernorm.weight")),
      q = nv(paste0(p, "self_attn.q_proj.weight"), TRUE),
      k = nv(paste0(p, "self_attn.k_proj.weight"), TRUE),
      v = nv(paste0(p, "self_attn.v_proj.weight"), TRUE),
      o = nv(paste0(p, "self_attn.o_proj.weight"), TRUE),
      post_ln = nv(paste0(p, "post_attention_layernorm.weight")),
      gate = nv(paste0(p, "mlp.gate_proj.weight"), TRUE),
      up = nv(paste0(p, "mlp.up_proj.weight"), TRUE),
      down = nv(paste0(p, "mlp.down_proj.weight"), TRUE)
    )
  })
  list(layers = layers, norm = nv("norm.weight"))
}

#' T3 Llama backbone forward (anvl, no cache)
#'
#' Torch-free port of \code{llama_model} given input embeddings: RoPE
#' (llama3-scaled, split-half) + causal SDPA + RMSNorm + SwiGLU, pre-norm,
#' 30 layers, final RMSNorm.
#'
#' @param inputs_embeds AnvlArray \code{[B, S, hidden]}.
#' @param w Weights from \code{\link{yq_llama_load_weights}}.
#' @param n_head,head_dim Attention shape (default 16 / 64).
#' @param eps RMSNorm epsilon.
#'
#' @return AnvlArray \code{[B, S, hidden]}.
#'
#' @export
yq_llama <- function(inputs_embeds, w, n_head = 16L, head_dim = 64L,
                     eps = 1e-5) {
  s <- anvl::shape(inputs_embeds)
  batch <- s[1L]
  seq <- s[2L]
  hidden <- s[3L]
  half <- head_dim %/% 2L

  rope <- .yq_llama_rope(seq, head_dim)
  bc <- function(m) anvl::nv_broadcast_to(
    anvl::nv_reshape(anvl::nv_array(m, dtype = "f32"), c(1L, 1L, seq, half)),
    c(batch, n_head, seq, half))
  cos <- bc(rope$cos)
  sin <- bc(rope$sin)

  mr <- matrix(0, seq, seq)
  mr[upper.tri(mr)] <- -1e9
  mask <- anvl::nv_array(array(mr, c(1L, 1L, seq, seq)), dtype = "f32")

  x <- inputs_embeds
  for (ly in w$layers) {
    h <- yunque::rms_norm(x, ly$input_ln, eps)
    q <- yunque::rope_split(.yq_heads(yunque::linear(h, ly$q), batch, n_head, head_dim), cos, sin)
    k <- yunque::rope_split(.yq_heads(yunque::linear(h, ly$k), batch, n_head, head_dim), cos, sin)
    v <- .yq_heads(yunque::linear(h, ly$v), batch, n_head, head_dim)
    a <- yunque::sdpa(q, k, v, mask = mask)
    a <- anvl::nv_reshape(anvl::nv_transpose(a, c(1L, 3L, 2L, 4L)),
      c(batch, seq, hidden))
    x <- x + yunque::linear(a, ly$o)

    h2 <- yunque::rms_norm(x, ly$post_ln, eps)
    mlp <- yunque::linear(
      yunque::silu(yunque::linear(h2, ly$gate)) * yunque::linear(h2, ly$up),
      ly$down)
    x <- x + mlp
  }
  yunque::rms_norm(x, w$norm, eps)
}
