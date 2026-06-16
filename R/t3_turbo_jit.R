# Turbo (GPT-2 backbone) T3 inference via a jit_compile'd TorchScript
# decode step - the turbo counterpart of t3_jit.R.
#
# Same idea as the Llama jit backend: the full per-token forward runs as
# ONE TorchScript function inside libtorch, so R only dispatches prefill,
# sampling, and the loop shell (~a dozen calls per token) instead of
# ~300 per-op lantern calls. The turbo backbone is GPT-2, not Llama, so
# the decode step differs:
#   - LayerNorm (mean + variance, weight AND bias), not RMSNorm
#   - one combined QKV projection (c_attn) with bias, not separate q/k/v
#   - GELU (tanh approximation), not SwiGLU
#   - biases on every projection
#   - absolute position embeddings (wpe), not rotary: gpt2_model$forward
#     adds them for the prefill, and this loop adds wpe(pos) to each decode
#     token's embedding before the step (the step itself takes no cos/sin).
# Correctness is checked against the eager gpt2_block forward in
# inst/tinytest (max logit diff ~1e-5).

# Session cache for compiled decode steps (keyed by architecture)
.jit_gpt2_decode_cache <- new.env(parent = emptyenv())

#' Build and compile the TorchScript GPT-2 decode step for an architecture
#'
#' @param n_layers,n_heads,head_dim,hidden,eps GPT-2 architecture params
#' @return The compiled script function
#' @noRd
.get_gpt2_jit_decode_step <- function(n_layers, n_heads, head_dim, hidden, eps) {
    key <- paste(n_layers, n_heads, head_dim, hidden, eps, sep = "_")
    if (!is.null(.jit_gpt2_decode_cache[[key]])) {
        return(.jit_gpt2_decode_cache[[key]])
    }

    # ATen-builtin calls only (same constraint as t3_jit.R): torch.matmul,
    # torch.rsqrt, torch.tanh, torch.scaled_dot_product_attention resolve;
    # the torch.nn.functional namespace does not, so LayerNorm and the
    # GELU-tanh activation are written out by hand. 12 weights per layer,
    # in .get_gpt2_layer_weights() order.
    src <- sprintf("
def decode_step(h: Tensor, w: List[Tensor], k_cache: Tensor, v_cache: Tensor,
                pos: int, valid: int) -> Tensor:
    n_layers = %d
    n_heads = %d
    head_dim = %d
    hidden = %d
    eps = %s
    gc0 = 0.7978845608028654
    gc1 = 0.044715
    B = h.size(0)
    for i in range(n_layers):
        b = i * 12
        resid = h
        mean = h.mean(-1, keepdim=True)
        centered = h - mean
        var = centered.pow(2).mean(-1, keepdim=True)
        normed = centered * torch.rsqrt(var + eps) * w[b] + w[b + 1]
        qkv = torch.matmul(normed, w[b + 2].t()) + w[b + 3]
        q = qkv[:, :, 0:hidden].view(B, 1, n_heads, head_dim).transpose(1, 2)
        k = qkv[:, :, hidden:hidden * 2].view(B, 1, n_heads, head_dim).transpose(1, 2)
        v = qkv[:, :, hidden * 2:hidden * 3].view(B, 1, n_heads, head_dim).transpose(1, 2)
        k_cache[i, :, :, pos] = k.squeeze(2)
        v_cache[i, :, :, pos] = v.squeeze(2)
        kv = k_cache[i, :, :, :valid]
        vv = v_cache[i, :, :, :valid]
        attn = torch.scaled_dot_product_attention(q, kv, vv)
        attn = attn.transpose(1, 2).reshape(B, 1, hidden)
        h = resid + (torch.matmul(attn, w[b + 4].t()) + w[b + 5])
        resid = h
        mean = h.mean(-1, keepdim=True)
        centered = h - mean
        var = centered.pow(2).mean(-1, keepdim=True)
        normed = centered * torch.rsqrt(var + eps) * w[b + 6] + w[b + 7]
        x = torch.matmul(normed, w[b + 8].t()) + w[b + 9]
        x = 0.5 * x * (1.0 + torch.tanh(gc0 * (x + gc1 * x * x * x)))
        h = resid + (torch.matmul(x, w[b + 10].t()) + w[b + 11])
    return h
"
                   , n_layers, n_heads, head_dim, hidden,
                   format(eps, scientific = FALSE))

    cu <- torch::jit_compile(src)
    .jit_gpt2_decode_cache[[key]] <- cu$decode_step
    cu$decode_step
}

#' Extract per-layer GPT-2 weight tensors in decode-step order
#'
#' Twelve tensors per layer: ln_1 weight/bias, c_attn weight/bias, c_proj
#' weight/bias, ln_2 weight/bias, c_fc weight/bias, mlp c_proj weight/bias.
#' Borrowed by reference - no copies.
#'
#' @param model T3 turbo model
#' @return Flat list of n_layers * 12 tensors
#' @noRd
.get_gpt2_layer_weights <- function(model) {
    n_layers <- model$tfmr$config$n_layer
    layers <- vector("list", n_layers)
    for (i in seq_len(n_layers)) {
        layer <- model$tfmr$h[[i]]
        layers[[i]] <- list(layer$ln_1$weight, layer$ln_1$bias,
                            layer$attn$c_attn$weight, layer$attn$c_attn$bias,
                            layer$attn$c_proj$weight, layer$attn$c_proj$bias,
                            layer$ln_2$weight, layer$ln_2$bias,
                            layer$mlp$c_fc$weight, layer$mlp$c_fc$bias,
                            layer$mlp$c_proj$weight, layer$mlp$c_proj$bias)
    }
    do.call(c, layers)
}

#' Turbo T3 inference with a TorchScript decode loop
#'
#' GPT-2 counterpart of \code{\link{t3_inference_jit}}. Runs prefill
#' eagerly, then executes each token's full GPT-2 forward as a single
#' jit_compile'd TorchScript call. No CFG (turbo), no rotary embeddings.
#' The KV cache auto-sizes (conditioning + max_new_tokens), so generation
#' always completes; the script compiles once per session.
#'
#' @param model T3 turbo model
#' @param cond T3 conditioning
#' @param text_tokens Tokenized text (tensor)
#' @param max_new_tokens Maximum speech tokens to generate
#' @param temperature Sampling temperature
#' @param top_k Top-k sampling
#' @param top_p Nucleus sampling threshold
#' @param repetition_penalty Repetition penalty
#' @param max_cache_len KV cache positions; NULL (default) auto-sizes
#' @return Generated speech tokens (0-indexed), with eos_found attribute
#' @export
t3_inference_turbo_jit <- function(model, cond, text_tokens,
                                   max_new_tokens = 1000, temperature = 0.8,
                                   top_k = 1000L, top_p = 0.95,
                                   repetition_penalty = 1.2,
                                   max_cache_len = NULL) {
    config <- model$config
    lcfg <- model$tfmr$config
    device <- model$speech_emb$weight$device
    n_layers <- lcfg$n_layer
    n_heads <- lcfg$n_head
    hidden <- lcfg$hidden_size
    head_dim <- hidden %/% n_heads

    step_fn <- .get_gpt2_jit_decode_step(n_layers, n_heads, head_dim, hidden,
                                         lcfg$layer_norm_epsilon)
    wflat <- .get_gpt2_layer_weights(model)

    if (text_tokens$dim() == 1L) {
        text_tokens <- text_tokens$unsqueeze(1L)
    }

    bos <- torch::torch_tensor(matrix(config$start_speech_token, nrow = 1L),
                               device = device, dtype = torch::torch_long())

    prep <- model$prepare_input_embeds(cond, text_tokens, bos)
    embeds <- prep$embeds
    cond_len <- embeds$size(2)
    batch <- embeds$size(1)

    if (is.null(max_cache_len)) {
        max_cache_len <- cond_len + max_new_tokens + 1L
    }
    # Clamp generation to the pre-allocated KV cache instead of indexing
    # past it (see t3_inference_jit).
    max_gen <- min(max_new_tokens, max_cache_len - cond_len - 1L)
    if (max_gen < 1L) {
        stop("Conditioning (", cond_len, " tokens) fills the KV cache (",
             max_cache_len, "); increase max_cache_len.", call. = FALSE)
    }
    if (max_gen < max_new_tokens) {
        warning("Capping generation at ", max_gen, " tokens (cache ",
                max_cache_len, ", conditioning ", cond_len, ").", call. = FALSE)
    }

    torch::with_no_grad({
        out <- model$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
        k_cache <- torch::torch_zeros(n_layers, batch, n_heads,
                                      max_cache_len, head_dim, device = device)
        v_cache <- torch::torch_zeros_like(k_cache)
        for (l in seq_len(n_layers)) {
            kv <- out$past_key_values[[l]]
            k_cache[l,,, 1:cond_len,] <- kv[[1]]
            v_cache[l,,, 1:cond_len,] <- kv[[2]]
        }
        h_last <- out$last_hidden_state[, -1,, drop = FALSE] # already ln_f'd

        generated_tokens <- list()
        eos_found <- FALSE
        last_token_id <- -1L
        repeat_run <- 0L

        for (i in seq_len(max_gen)) {
            logits <- model$speech_head$forward(h_last)$squeeze(2L)
            # repetition-penalty set: BOS + generated so far (1-indexed),
            # matching t3_inference_turbo
            gen_ids <- torch::torch_cat(
                c(list(bos$add(1L)), generated_tokens), dim = 2L)
            next_token <- .turbo_sample_token(logits, gen_ids, temperature,
                                              top_k, top_p, repetition_penalty)
            generated_tokens[[length(generated_tokens) + 1L]] <- next_token

            token_id <- as.integer(next_token$cpu()) - 1L
            if (token_id == config$stop_speech_token) {
                message("EOS at step ", i)
                eos_found <- TRUE
                break
            }

            # Runaway guard (see t3_inference)
            if (token_id == last_token_id) {
                repeat_run <- repeat_run + 1L
                if (repeat_run >= 10L) {
                    warning("Stopping generation: token ", token_id,
                            " repeated 10x at step ", i, " (degenerate loop)",
                            call. = FALSE)
                    break
                }
            } else {
                last_token_id <- token_id
                repeat_run <- 1L
            }

            pos0 <- cond_len + i - 1L                   # 0-based absolute slot
            # wpe at this absolute position (the step_fn bypasses the
            # forward that would otherwise add it, as HF GPT2Model does)
            pos_emb <- model$tfmr$wpe$forward(
                torch::torch_tensor(matrix(pos0, nrow = 1L), device = device,
                                    dtype = torch::torch_long())$add(1L))
            emb <- model$speech_emb$forward(next_token) + pos_emb
            h <- step_fn(emb, wflat, k_cache, v_cache,
                         torch::jit_scalar(pos0), torch::jit_scalar(pos0 + 1L))
            h_last <- model$tfmr$ln_f$forward(h)
        }
    })

    if (length(generated_tokens) > 0L) {
        tokens <- torch::torch_cat(generated_tokens, dim = 2L)$squeeze(1L)
        tokens <- tokens$sub(1L)
        token_vals <- as.integer(tokens$cpu())
        if (length(token_vals) > 0L &&
            token_vals[length(token_vals)] == config$stop_speech_token) {
            tokens <- tokens[1:(tokens$size(1) - 1L)]
        }
    } else {
        tokens <- torch::torch_tensor(integer(0), device = device)
    }
    attr(tokens, "eos_found") <- eos_found
    tokens
}
