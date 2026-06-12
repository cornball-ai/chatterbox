# The TorchScript decode step must compile and agree with the same
# computation done eagerly in R. Runs on CPU with a tiny fake
# architecture - no model weights needed.

if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    n_layers <- 2L; n_heads <- 2L; head_dim <- 4L
    hidden <- n_heads * head_dim
    eps <- 1e-5
    torch::torch_manual_seed(7)

    step_fn <- chatterbox:::.get_jit_decode_step(n_layers, n_heads, head_dim, eps)

    # Random weights in extractor order (ln, q, k, v, o, post_ln, gate, up, down)
    mk <- function (out, inn) torch::torch_randn(out, inn) * 0.1
    w <- list()
    for (i in seq_len(n_layers)) {
        w <- c(w, list(
            torch::torch_ones(hidden), mk(hidden, hidden), mk(hidden, hidden),
            mk(hidden, hidden), mk(hidden, hidden), torch::torch_ones(hidden),
            mk(hidden * 2L, hidden), mk(hidden * 2L, hidden), mk(hidden, hidden * 2L)
        ))
    }

    B <- 2L; cache_len <- 6L; pos0 <- 3L # 3 positions already filled
    h0 <- torch::torch_randn(B, 1L, hidden)
    k_cache <- torch::torch_randn(n_layers, B, n_heads, cache_len, head_dim)
    v_cache <- torch::torch_randn(n_layers, B, n_heads, cache_len, head_dim)
    cos <- torch::torch_randn(1L, 1L, 1L, head_dim)
    sin <- torch::torch_randn(1L, 1L, 1L, head_dim)

    # Eager reference: identical math in plain R torch
    rms <- function (x, wt) {
        xf <- x$to(dtype = torch::torch_float32())
        v <- xf$pow(2)$mean(dim = -1, keepdim = TRUE)
        (wt * (xf * torch::torch_rsqrt(v + eps)))$to(dtype = x$dtype)
    }
    rot <- function (x) {
        half <- head_dim %/% 2L
        torch::torch_cat(list(-x[, , , (half + 1L):head_dim], x[, , , 1:half]), dim = -1L)
    }
    kc <- k_cache$clone(); vc <- v_cache$clone()
    h <- h0
    for (i in seq_len(n_layers)) {
        b <- (i - 1L) * 9L
        resid <- h
        nx <- rms(h, w[[b + 1L]])
        q <- torch::torch_matmul(nx, w[[b + 2L]]$t())$view(c(B, 1L, n_heads, head_dim))$transpose(2L, 3L)
        k <- torch::torch_matmul(nx, w[[b + 3L]]$t())$view(c(B, 1L, n_heads, head_dim))$transpose(2L, 3L)
        v <- torch::torch_matmul(nx, w[[b + 4L]]$t())$view(c(B, 1L, n_heads, head_dim))$transpose(2L, 3L)
        q <- q * cos + rot(q) * sin
        k <- k * cos + rot(k) * sin
        kc[i, , , pos0 + 1L, ] <- k$squeeze(3)
        vc[i, , , pos0 + 1L, ] <- v$squeeze(3)
        att <- torch::torch_scaled_dot_product_attention(
            q, kc[i, , , 1:(pos0 + 1L), ], vc[i, , , 1:(pos0 + 1L), ])
        att <- att$transpose(2L, 3L)$reshape(c(B, 1L, hidden))
        h <- resid + torch::torch_matmul(att, w[[b + 5L]]$t())
        resid <- h
        nx <- rms(h, w[[b + 6L]])
        gate <- torch::torch_matmul(nx, w[[b + 7L]]$t())
        up <- torch::torch_matmul(nx, w[[b + 8L]]$t())
        h <- resid + torch::torch_matmul(torch::nnf_silu(gate) * up, w[[b + 9L]]$t())
    }

    out <- step_fn(h0, w, k_cache, v_cache, cos, sin,
        torch::jit_scalar(pos0), torch::jit_scalar(pos0 + 1L))

    expect_true(as.numeric(torch::torch_max(torch::torch_abs(out - h))$cpu()) < 1e-5)
    # In-place cache write crossed the script boundary
    expect_true(as.numeric(torch::torch_max(torch::torch_abs(
        k_cache[1, , , pos0 + 1L, ] - kc[1, , , pos0 + 1L, ]))$cpu()) < 1e-5)
}
