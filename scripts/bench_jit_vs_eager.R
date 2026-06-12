# Documentation benchmark: where does the speed come from?
#   pure-R     : generate(backend = "r") - nn_module forward path
#   lean-eager : same decode math in plain R, ATen builtins only
#                (torch_matmul/.t(), torch_scaled_dot_product_attention,
#                nnf_silu) - no nn_module dispatch, still one R call per op
#   jit        : generate(backend = "jit") - whole step in TorchScript
# Long text (~500+ tokens), tuned GC, T3 stage isolated for the manual
# variants; jit/pure-R measured end-to-end via the public API too.

options(torch.cuda_allocator_reserved_rate = 0.5, torch.threshold_call_gc = 16000)
library(chatterbox)
library(torch)

long <- "Yes, Rarely or never Almost never, at most once in a while, over the past week Sometimes Only a couple of days over the past week, not many times in any given day Often Four or more days over the past week, several times each day Very Often just about every day over the past week, multiple times throughout the Day."
model <- load_chatterbox(chatterbox("cuda"))
voice <- create_voice_embedding(model, system.file("audio", "jfk.wav", package = "chatterbox"))

t3 <- model$t3
lcfg <- t3$tfmr$config
n_layers <- lcfg$num_hidden_layers
n_heads <- lcfg$num_attention_heads
head_dim <- lcfg$head_dim
eps <- lcfg$rms_norm_eps
hidden <- n_heads * head_dim

txt <- chatterbox:::punc_norm(normalize_tts_text(long))
tt <- torch_tensor(matrix(chatterbox:::tokenize_text(model$tokenizer, txt), nrow = 1),
    dtype = torch_long())$to(device = model$device)
cond <- chatterbox:::t3_cond(speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens, emotion_adv = 0.5)

stage <- function (label, fn) {
    for (r in 1:2) {
        t0 <- Sys.time()
        toks <- with_no_grad(fn())
        s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        n <- as.integer(toks$size(1))
        cat(sprintf("%-12s run %d: %5.1fs, %4d tokens, %4.0f ms/tok, eos=%s\n",
            label, r, s, n, 1000 * s / n, isTRUE(attr(toks, "eos_found"))))
        gc(verbose = FALSE)
    }
}

# --- 1. pure R (nn_module path), T3 stage ---
stage("pure-R", function () chatterbox:::t3_inference(t3, cond, tt))

# --- 2. jit backend, T3 stage ---
stage("jit", function () t3_inference_jit(t3, cond, tt))

# --- 3. lean-eager: ATen-builtin R, structurally identical to the script ---
wflat <- chatterbox:::.get_layer_weights(t3)
lean_step <- function (h, k_cache, v_cache, cosp, sinp, pos1, valid) {
    B <- h$size(1)
    for (i in seq_len(n_layers)) {
        b <- (i - 1L) * 9L
        resid <- h
        hf <- h$to(dtype = torch_float32())
        v_ <- hf$pow(2)$mean(dim = -1, keepdim = TRUE)
        nx <- (wflat[[b + 1L]] * (hf * torch_rsqrt(v_ + eps)))$to(dtype = h$dtype)
        q <- torch_matmul(nx, wflat[[b + 2L]]$t())$view(c(B, 1L, n_heads, head_dim))$transpose(2L, 3L)
        k <- torch_matmul(nx, wflat[[b + 3L]]$t())$view(c(B, 1L, n_heads, head_dim))$transpose(2L, 3L)
        v <- torch_matmul(nx, wflat[[b + 4L]]$t())$view(c(B, 1L, n_heads, head_dim))$transpose(2L, 3L)
        half <- head_dim %/% 2L
        qr <- torch_cat(list(-q[, , , (half + 1L):head_dim], q[, , , 1:half]), dim = -1L)
        kr <- torch_cat(list(-k[, , , (half + 1L):head_dim], k[, , , 1:half]), dim = -1L)
        q <- q * cosp + qr * sinp
        k <- k * cosp + kr * sinp
        k_cache[i, , , pos1, ] <- k$squeeze(3)
        v_cache[i, , , pos1, ] <- v$squeeze(3)
        att <- torch_scaled_dot_product_attention(
            q, k_cache[i, , , 1:valid, ], v_cache[i, , , 1:valid, ])
        att <- att$transpose(2L, 3L)$reshape(c(B, 1L, hidden))
        h <- resid + torch_matmul(att, wflat[[b + 5L]]$t())
        resid <- h
        hf <- h$to(dtype = torch_float32())
        v_ <- hf$pow(2)$mean(dim = -1, keepdim = TRUE)
        nx <- (wflat[[b + 6L]] * (hf * torch_rsqrt(v_ + eps)))$to(dtype = h$dtype)
        gate <- torch_matmul(nx, wflat[[b + 7L]]$t())
        up <- torch_matmul(nx, wflat[[b + 8L]]$t())
        h <- resid + torch_matmul(nnf_silu(gate) * up, wflat[[b + 9L]]$t())
    }
    h
}

lean_infer <- function () {
    config <- t3$config
    device <- t3$text_emb$weight$device
    text_tokens <- nnf_pad(nnf_pad(tt, c(1, 0), value = config$start_text_token),
        c(0, 1), value = config$stop_text_token)
    text_tokens <- torch_cat(list(text_tokens, text_tokens), dim = 1)
    bos <- torch_tensor(matrix(config$start_speech_token, nrow = 1),
        device = device, dtype = torch_long())
    bos2 <- torch_cat(list(bos, bos), dim = 1)
    prep <- t3$prepare_input_embeds(cond, text_tokens, bos2, 0.5)
    be <- t3$speech_emb$forward(bos$add(1L)) + t3$speech_pos_emb$get_fixed_embedding(0)
    embeds <- torch_cat(list(prep$embeds, torch_cat(list(be, be), dim = 1)), dim = 2)
    cond_len <- embeds$size(2)
    cache_len <- cond_len + 1001L
    rope <- chatterbox:::compute_rope_frequencies(head_dim, cache_len + 100L,
        theta = lcfg$rope_theta, scaling = lcfg$rope_scaling, device = device)
    out <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
    k_cache <- torch_zeros(n_layers, 2L, n_heads, cache_len, head_dim, device = device)
    v_cache <- torch_zeros_like(k_cache)
    for (l in seq_len(n_layers)) {
        kv <- out$past_key_values[[l]]
        k_cache[l, , , 1:cond_len, ] <- kv[[1]]
        v_cache[l, , , 1:cond_len, ] <- kv[[2]]
    }
    h_last <- out$last_hidden_state[, -1, , drop = FALSE]
    generated <- bos$add(1L)
    predicted <- list()
    eos_found <- FALSE
    last_id <- -1L; run_n <- 0L
    for (i in seq_len(1000L)) {
        logits <- t3$speech_head$forward(h_last)$squeeze(2)
        lg <- logits[1, ]$unsqueeze(1) + 0.5 * (logits[1, ]$unsqueeze(1) - logits[2, ]$unsqueeze(1))
        nt <- chatterbox:::.sample_speech_token(lg, generated, 0.8, 1.0, 0.05, 1.2)
        predicted[[i]] <- nt
        generated <- torch_cat(list(generated, nt), dim = 2)
        tid <- as.integer(nt$cpu()) - 1L
        if (tid == config$stop_speech_token) { eos_found <- TRUE; break }
        if (tid == last_id) { run_n <- run_n + 1L; if (run_n >= 10L) break } else { last_id <- tid; run_n <- 1L }
        emb <- t3$speech_emb$forward(nt) + t3$speech_pos_emb$get_fixed_embedding(i)
        emb <- torch_cat(list(emb, emb), dim = 1)
        pos0 <- cond_len + i - 1L
        cosp <- rope$cos[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
        sinp <- rope$sin[pos0 + 1L, ]$view(c(1L, 1L, 1L, head_dim))
        h <- lean_step(emb, k_cache, v_cache, cosp, sinp, pos0 + 1L, pos0 + 1L)
        h_last <- t3$tfmr$norm$forward(h)
    }
    toks <- torch_cat(predicted, dim = 2)$squeeze(1)$sub(1L)
    attr(toks, "eos_found") <- eos_found
    toks
}
stage("lean-eager", lean_infer)
