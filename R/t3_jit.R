# T3 inference via a jit_compile'd TorchScript decode step.
#
# The full 30-layer per-token forward runs as ONE TorchScript function
# inside libtorch; R keeps prefill, sampling, and the loop shell
# (~15 dispatched R calls per token). Measured long-form, same text,
# tuned GC (scripts/bench_jit_vs_eager.R):
#   pure R (nn_module path)              87-88 ms/token
#   lean eager R (ATen builtins, no
#     nn_module/nnf dispatch)            71 ms/token
#   this backend                         11 ms/token
#   retired C++ backend (reference)       9 ms/token
# The lean-eager row is why this exists: even optimally-written eager R
# keeps a ~70 ms/token floor - the cost is the per-op R->lantern call
# itself (~190us x ~370 ops/token), not wrapper style. Only removing
# the per-op R call wins, and TorchScript does it without compiled
# code, configure scripts, or linking against torch's private
# libraries. Correctness vs the eager forward: max diff 2e-6.

# Session cache for compiled decode steps (keyed by architecture)
.jit_decode_cache <- new.env(parent = emptyenv())

#' Build and compile the TorchScript decode step for an architecture
#'
#' @param n_layers,n_heads,head_dim,eps Llama architecture parameters
#' @return The compiled script function
#' @noRd
.get_jit_decode_step <- function (n_layers, n_heads, head_dim, eps) {
    key <- paste(n_layers, n_heads, head_dim, eps, sep = "_")
    if (!is.null(.jit_decode_cache[[key]])) {
        return(.jit_decode_cache[[key]])
    }

    # ATen-builtin calls only: this lantern's TorchScript environment
    # resolves torch.matmul/torch.silu/torch.scaled_dot_product_attention
    # but not the torch.nn.functional namespace or dtype constants
    src <- sprintf("
def decode_step(h: Tensor, w: List[Tensor], k_cache: Tensor, v_cache: Tensor,
                cos: Tensor, sin: Tensor, pos: int, valid: int) -> Tensor:
    n_layers = %d
    n_heads = %d
    head_dim = %d
    eps = %s
    half = head_dim // 2
    B = h.size(0)
    for i in range(n_layers):
        b = i * 9
        resid = h
        hf = h.float()
        var = hf.pow(2).mean(-1, keepdim=True)
        normed = (w[b] * (hf * torch.rsqrt(var + eps))).to(h.dtype)
        q = torch.matmul(normed, w[b + 1].t()).view(B, 1, n_heads, head_dim).transpose(1, 2)
        k = torch.matmul(normed, w[b + 2].t()).view(B, 1, n_heads, head_dim).transpose(1, 2)
        v = torch.matmul(normed, w[b + 3].t()).view(B, 1, n_heads, head_dim).transpose(1, 2)
        q = q * cos + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * sin
        k = k * cos + torch.cat([-k[..., half:], k[..., :half]], dim=-1) * sin
        k_cache[i, :, :, pos] = k.squeeze(2)
        v_cache[i, :, :, pos] = v.squeeze(2)
        kv = k_cache[i, :, :, :valid]
        vv = v_cache[i, :, :, :valid]
        attn = torch.scaled_dot_product_attention(q, kv, vv)
        attn = attn.transpose(1, 2).reshape(B, 1, n_heads * head_dim)
        h = resid + torch.matmul(attn, w[b + 4].t())
        resid = h
        hf = h.float()
        var = hf.pow(2).mean(-1, keepdim=True)
        normed = (w[b + 5] * (hf * torch.rsqrt(var + eps))).to(h.dtype)
        gate = torch.matmul(normed, w[b + 6].t())
        up = torch.matmul(normed, w[b + 7].t())
        h = resid + torch.matmul(torch.silu(gate) * up, w[b + 8].t())
    return h
", n_layers, n_heads, head_dim, format(eps, scientific = FALSE))

    cu <- torch::jit_compile(src)
    .jit_decode_cache[[key]] <- cu$decode_step
    cu$decode_step
}

#' Extract per-layer weight tensors in decode-step order
#'
#' Nine tensors per layer: input_layernorm, q/k/v/o projections,
#' post_attention_layernorm, gate/up/down projections. Tensors are
#' borrowed by reference - no copies.
#'
#' @param model T3 model
#' @return Flat list of n_layers * 9 tensors
#' @noRd
.get_layer_weights <- function (model) {
    n_layers <- model$tfmr$config$num_hidden_layers
    layers <- vector("list", n_layers)
    for (i in seq_len(n_layers)) {
        layer <- model$tfmr$layers[[i]]
        layers[[i]] <- list(layer$input_layernorm$weight,
                            layer$self_attn$q_proj$weight,
                            layer$self_attn$k_proj$weight,
                            layer$self_attn$v_proj$weight,
                            layer$self_attn$o_proj$weight,
                            layer$post_attention_layernorm$weight,
                            layer$mlp$gate_proj$weight,
                            layer$mlp$up_proj$weight,
                            layer$mlp$down_proj$weight)
    }
    do.call(c, layers)
}

#' T3 inference with a TorchScript decode loop
#'
#' Runs prefill eagerly, then executes each token's 30-layer forward as
#' a single jit_compile'd TorchScript call. The KV cache is auto-sized
#' (conditioning + max_new_tokens), so generation always completes;
#' there is no shape freezing and no per-size recompilation - the
#' script compiles once per session in milliseconds.
#'
#' @param model T3 model
#' @param cond T3 conditioning
#' @param text_tokens Tokenized text (tensor)
#' @param max_new_tokens Maximum speech tokens to generate
#' @param temperature Sampling temperature
#' @param cfg_weight Classifier-free guidance weight
#' @param top_p Nucleus sampling threshold (1.0 disables, Python default)
#' @param min_p Minimum probability threshold
#' @param repetition_penalty Repetition penalty
#' @param max_cache_len KV cache positions; NULL (default) auto-sizes
#' @return Generated speech tokens (0-indexed), with eos_found attribute
#' @export
t3_inference_jit <- function (model, cond, text_tokens, max_new_tokens = 1000,
                              temperature = 0.8, cfg_weight = 0.5, top_p = 1.0,
                              min_p = 0.05, repetition_penalty = 1.2,
                              max_cache_len = NULL) {
    config <- model$config
    lcfg <- model$tfmr$config
    device <- model$text_emb$weight$device
    n_layers <- lcfg$num_hidden_layers
    n_heads <- lcfg$num_attention_heads
    head_dim <- lcfg$head_dim

    step_fn <- .get_jit_decode_step(n_layers, n_heads, head_dim,
        lcfg$rms_norm_eps)
    wflat <- .get_layer_weights(model)

    if (text_tokens$dim() == 1) {
        text_tokens <- text_tokens$unsqueeze(1)
    }
    text_tokens <- torch::nnf_pad(text_tokens, c(1, 0),
        value = config$start_text_token)
    text_tokens <- torch::nnf_pad(text_tokens, c(0, 1),
        value = config$stop_text_token)
    use_cfg <- cfg_weight > 0.0
    if (use_cfg) {
        text_tokens <- torch::torch_cat(list(text_tokens, text_tokens), dim = 1)
    }

    bos_token <- torch::torch_tensor(
        matrix(config$start_speech_token, nrow = 1),
        device = device, dtype = torch::torch_long()
    )
    bos_in <- if (use_cfg) {
        torch::torch_cat(list(bos_token, bos_token), dim = 1)
    } else {
        bos_token
    }

    # Prefill context + second BOS frame (Python parity; see t3_inference)
    prep <- model$prepare_input_embeds(cond, text_tokens, bos_in, cfg_weight)
    embeds <- prep$embeds
    bos_emb <- model$speech_emb$forward(bos_token$add(1L)) +
        model$speech_pos_emb$get_fixed_embedding(0)
    if (use_cfg) {
        bos_emb <- torch::torch_cat(list(bos_emb, bos_emb), dim = 1)
    }
    embeds <- torch::torch_cat(list(embeds, bos_emb), dim = 2)
    cond_len <- embeds$size(2)
    batch <- embeds$size(1)

    if (is.null(max_cache_len)) {
        max_cache_len <- cond_len + max_new_tokens + 1L
    }
    rope <- compute_rope_frequencies(head_dim, max_cache_len + 100L,
        theta = lcfg$rope_theta, scaling = lcfg$rope_scaling, device = device)

    torch::with_no_grad({
        out <- model$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
        k_cache <- torch::torch_zeros(n_layers, batch, n_heads,
            max_cache_len, head_dim, device = device)
        v_cache <- torch::torch_zeros_like(k_cache)
        for (l in seq_len(n_layers)) {
            kv <- out$past_key_values[[l]]
            k_cache[l, , , 1:cond_len, ] <- kv[[1]]
            v_cache[l, , , 1:cond_len, ] <- kv[[2]]
        }
        h_last <- out$last_hidden_state[, -1, , drop = FALSE]

        # 1-indexed throughout (see t3_inference); penalty set includes BOS
        generated_ids <- bos_token$add(1L)
        predicted <- list()
        eos_found <- FALSE
        last_token_id <- -1L
        repeat_run <- 0L

        for (i in seq_len(max_new_tokens)) {
            logits <- model$speech_head$forward(h_last)$squeeze(2)
            if (use_cfg) {
                cond_logits <- logits[1, ]$unsqueeze(1)
                uncond_logits <- logits[2, ]$unsqueeze(1)
                logits <- cond_logits + cfg_weight * (cond_logits - uncond_logits)
            } else {
                logits <- logits[1, ]$unsqueeze(1)
            }

            next_token <- .sample_speech_token(logits, generated_ids,
                temperature, top_p, min_p, repetition_penalty)
            predicted[[length(predicted) + 1]] <- next_token
            generated_ids <- torch::torch_cat(list(generated_ids, next_token),
                dim = 2)

            token_id <- as.integer(next_token$cpu()) - 1L
            if (token_id == config$stop_speech_token) {
                message("EOS detected at step ", i)
                eos_found <- TRUE
                break
            }

            # Runaway guard (see t3_inference)
            if (token_id == last_token_id) {
                repeat_run <- repeat_run + 1L
                if (repeat_run >= 10L) {
                    warning("Stopping generation: token ", token_id,
                        " repeated 10x at step ", i,
                        " (degenerate loop)", call. = FALSE)
                    break
                }
            } else {
                last_token_id <- token_id
                repeat_run <- 1L
            }

            emb <- model$speech_emb$forward(next_token) +
                model$speech_pos_emb$get_fixed_embedding(i)
            if (use_cfg) {
                emb <- torch::torch_cat(list(emb, emb), dim = 1)
            }

            pos0 <- cond_len + i - 1L # 0-based absolute position
            cosp <- rope$cos[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
            sinp <- rope$sin[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
            h <- step_fn(emb, wflat, k_cache, v_cache, cosp, sinp,
                torch::jit_scalar(pos0), torch::jit_scalar(pos0 + 1L))
            h_last <- model$tfmr$norm$forward(h)
        }
    })

    if (length(predicted) > 0) {
        tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)
        tokens <- tokens$sub(1L)
    } else {
        tokens <- torch::torch_tensor(integer(0), device = device)
    }
    attr(tokens, "eos_found") <- eos_found
    tokens
}
