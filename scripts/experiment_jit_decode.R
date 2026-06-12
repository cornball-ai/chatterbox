# EXPERIMENT: TorchScript decode loop via jit_compile - can it match the
# C++ backend (9 ms/token long-form) without shipping compiled code?
# The scripted function runs all 30 decoder layers per token inside
# libtorch; R keeps only prefill, sampling, and the loop shell.

options(torch.cuda_allocator_reserved_rate = 0.5, torch.threshold_call_gc = 16000)
library(chatterbox)
library(torch)

long <- "Yes, Rarely or never Almost never, at most once in a while, over the past week Sometimes Only a couple of days over the past week, not many times in any given day Often Four or more days over the past week, several times each day Very Often just about every day over the past week, multiple times throughout the Day."

model <- load_chatterbox(chatterbox("cuda"))
voice <- create_voice_embedding(model, system.file("audio", "jfk.wav", package = "chatterbox"))

t3 <- model$t3
config <- t3$config
lcfg <- t3$tfmr$config
n_layers <- lcfg$num_hidden_layers
n_heads <- lcfg$num_attention_heads
head_dim <- lcfg$head_dim
eps <- lcfg$rms_norm_eps

# --- TorchScript: one full token step through all layers ---
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
cu <- jit_compile(src)
cat("script compiled\n")

# --- weights, flat (reuse the cpp extractor's per-layer 9-tensor order) ---
wx <- chatterbox:::.get_cpp_weights(t3)
wflat <- do.call(c, wx$layers)
stopifnot(length(wflat) == n_layers * 9L)

# --- prefill (eager), then copy past_key_values into stacked caches ---
txt <- chatterbox:::punc_norm(normalize_tts_text(long))
tt <- torch_tensor(matrix(chatterbox:::tokenize_text(model$tokenizer, txt), nrow = 1),
    dtype = torch_long())$to(device = model$device)
cond <- chatterbox:::t3_cond(speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens, emotion_adv = 0.5)

sot <- config$start_text_token; eot <- config$stop_text_token
text_tokens <- nnf_pad(nnf_pad(tt, c(1, 0), value = sot), c(0, 1), value = eot)
text_tokens <- torch_cat(list(text_tokens, text_tokens), dim = 1)
bos <- torch_tensor(matrix(config$start_speech_token, nrow = 1),
    device = model$device, dtype = torch_long())
bos2 <- torch_cat(list(bos, bos), dim = 1)
prep <- t3$prepare_input_embeds(cond, text_tokens, bos2, 0.5)
bos_emb <- t3$speech_emb$forward(bos$add(1L)) + t3$speech_pos_emb$get_fixed_embedding(0)
embeds <- torch_cat(list(prep$embeds, torch_cat(list(bos_emb, bos_emb), dim = 1)), dim = 2)
cond_len <- embeds$size(2)

max_new <- 1000L
cache_len <- cond_len + max_new + 1L
rope <- chatterbox:::compute_rope_frequencies(head_dim, cache_len + 100L,
    theta = lcfg$rope_theta, scaling = lcfg$rope_scaling, device = model$device)

run_jit <- function () {
    with_no_grad({
        out <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
        k_cache <- torch_zeros(n_layers, 2L, n_heads, cache_len, head_dim, device = model$device)
        v_cache <- torch_zeros_like(k_cache)
        for (l in seq_len(n_layers)) {
            kv <- out$past_key_values[[l]]
            k_cache[l, , , 1:cond_len, ] <- kv[[1]]
            v_cache[l, , , 1:cond_len, ] <- kv[[2]]
        }
        h_last <- out$last_hidden_state[, -1, , drop = FALSE]

        generated <- torch_tensor(matrix(config$start_speech_token + 1L, nrow = 1),
            device = model$device, dtype = torch_long())
        predicted <- list()
        eos_found <- FALSE
        last_id <- -1L; run_n <- 0L

        for (i in seq_len(max_new)) {
            logits <- t3$speech_head$forward(h_last)$squeeze(2)
            cl <- logits[1, ]$unsqueeze(1)
            ul <- logits[2, ]$unsqueeze(1)
            lg <- cl + 0.5 * (cl - ul)
            next_tok <- chatterbox:::.sample_speech_token(lg, generated, 0.8, 1.0, 0.05, 1.2)
            predicted[[i]] <- next_tok
            generated <- torch_cat(list(generated, next_tok), dim = 2)
            tid <- as.integer(next_tok$cpu()) - 1L
            if (tid == config$stop_speech_token) { eos_found <- TRUE; break }
            if (tid == last_id) { run_n <- run_n + 1L; if (run_n >= 10L) break } else { last_id <- tid; run_n <- 1L }

            emb <- t3$speech_emb$forward(next_tok) + t3$speech_pos_emb$get_fixed_embedding(i)
            emb <- torch_cat(list(emb, emb), dim = 1)
            pos0 <- cond_len + i - 1L            # 0-based absolute position
            cosp <- rope$cos[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
            sinp <- rope$sin[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
            h <- cu$decode_step(emb, wflat, k_cache, v_cache, cosp, sinp,
                jit_scalar(pos0), jit_scalar(pos0 + 1L))
            h_last <- t3$tfmr$norm$forward(h)
        }
        toks <- torch_cat(predicted, dim = 2)$squeeze(1)$sub(1L)
        attr(toks, "eos_found") <- eos_found
        toks
    })
}

# correctness probe: one scripted step vs eager continuation
with_no_grad({
    out <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
    k_cache <- torch_zeros(n_layers, 2L, n_heads, cache_len, head_dim, device = model$device)
    v_cache <- torch_zeros_like(k_cache)
    for (l in seq_len(n_layers)) {
        kv <- out$past_key_values[[l]]
        k_cache[l, , , 1:cond_len, ] <- kv[[1]]
        v_cache[l, , , 1:cond_len, ] <- kv[[2]]
    }
    probe_emb <- t3$speech_emb$forward(bos2$add(1L)) + t3$speech_pos_emb$get_fixed_embedding(1)
    eager <- t3$tfmr$forward(inputs_embeds = probe_emb, past_key_values = out$past_key_values,
        use_cache = TRUE)$last_hidden_state
    pos0 <- cond_len
    cosp <- rope$cos[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
    sinp <- rope$sin[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
    h <- cu$decode_step(probe_emb, wflat, k_cache, v_cache, cosp, sinp,
        jit_scalar(pos0), jit_scalar(pos0 + 1L))
    hn <- t3$tfmr$norm$forward(h)
    cat(sprintf("correctness: max diff scripted vs eager = %.6f\n",
        as.numeric(torch_max(torch_abs(hn - eager))$cpu())))
})

# benchmark
for (r in 1:2) {
    t0 <- Sys.time()
    toks <- run_jit()
    s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    n <- as.integer(toks$size(1))
    cat(sprintf("jit-decode-%d: %.1fs, %d tokens, %.0f ms/tok, eos=%s\n",
        r, s, n, 1000 * s / n, isTRUE(attr(toks, "eos_found"))))
    gc(verbose = FALSE)
}
