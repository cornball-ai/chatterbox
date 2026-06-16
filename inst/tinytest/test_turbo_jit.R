# The turbo GPT-2 jit decode step must match the eager gpt2 forward.
# Heavy (loads the turbo model) and needs the turbo weights downloaded,
# so home-only and skipped when turbo is absent.
library(chatterbox)

if (at_home() &&
    requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed() &&
    turbo_models_available()) {

    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
    m <- chatterbox(device, turbo = TRUE)
    t3 <- m$t3
    lcfg <- t3$tfmr$config
    nL <- lcfg$n_layer
    nH <- lcfg$n_head
    hid <- lcfg$hidden_size
    hd <- hid %/% nH

    torch::torch_manual_seed(0L)
    torch::with_no_grad({
        # Random prefill (real weights keep activations well-scaled; only
        # the decode-step math is under test, not the semantics).
        L <- 12L
        embeds <- torch::torch_randn(c(1L, L, hid), device = device)
        out <- t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
        pkv <- out$past_key_values

        h <- torch::torch_randn(c(1L, 1L, hid), device = device)

        # Eager: one decode token through the gpt2 forward with KV cache
        oe <- t3$tfmr$forward(inputs_embeds = h, past_key_values = pkv,
                              use_cache = TRUE)
        h_eager <- oe$last_hidden_state

        # JIT: same token through the TorchScript decode step + ln_f
        step <- chatterbox:::.get_gpt2_jit_decode_step(
            nL, nH, hd, hid, lcfg$layer_norm_epsilon)
        wf <- chatterbox:::.get_gpt2_layer_weights(t3)
        mcl <- L + 4L
        kc <- torch::torch_zeros(nL, 1L, nH, mcl, hd, device = device)
        vc <- torch::torch_zeros_like(kc)
        for (l in seq_len(nL)) {
            kv <- pkv[[l]]
            kc[l,,, 1:L,] <- kv[[1]]
            vc[l,,, 1:L,] <- kv[[2]]
        }
        # The eager forward adds wpe(position) internally; the step does
        # not, so add it to the decode token's embedding here too (as
        # t3_inference_turbo_jit does). Position = prefill length L.
        wp <- t3$tfmr$wpe$forward(torch::torch_tensor(matrix(L, nrow = 1L),
            device = device, dtype = torch::torch_long())$add(1L))
        hj <- step(h + wp, wf, kc, vc, torch::jit_scalar(L),
                   torch::jit_scalar(L + 1L))
        h_jit <- t3$tfmr$ln_f$forward(hj)

        diff <- as.numeric((h_eager - h_jit)$abs()$max()$cpu())
        # Correct math agrees to ~1e-5; a wrong step diverges by O(1).
        expect_true(diff < 1e-3,
                    info = paste("turbo jit decode hidden max abs diff:", diff))
    })
}
