# Sampling semantics for the shared T3 token sampler
# (HF logits-processor parity: sign-dependent repetition penalty,
#  top-p keeps the threshold-crossing token)

if (requireNamespace("torch", quietly = TRUE)) {
    sample_tok <- chatterbox:::.sample_speech_token

    mk_logits <- function (vals) {
        torch::torch_tensor(matrix(vals, nrow = 1), dtype = torch::torch_float())
    }
    no_ids <- torch::torch_tensor(matrix(integer(0), nrow = 1),
        dtype = torch::torch_long())

    torch::torch_manual_seed(42)

    # Unpenalized: overwhelming logit wins
    logits <- mk_logits(c(10, -10, -10, -10))
    tok <- sample_tok(logits$clone(), no_ids, 1.0, 1.0, 0.0, 1.0)
    expect_equal(as.integer(tok$cpu()), 1L)

    # Sign-dependent penalty on a NEGATIVE logit: multiply, don't divide.
    # Token 1 (logit -5, penalized by 10) must lose to token 2 (logit -10):
    # correct semantics give -50 < -10; the old divide bug gave -0.5 > -10.
    ids1 <- torch::torch_tensor(matrix(1L, nrow = 1), dtype = torch::torch_long())
    logits <- mk_logits(c(-5, -10, -1e4, -1e4))
    tok <- sample_tok(logits$clone(), ids1, 1.0, 1.0, 0.0, 10.0)
    expect_equal(as.integer(tok$cpu()), 2L)

    # Positive logit is divided: token 1 (logit 20, penalized by 10 -> 2)
    # must lose to token 2 (logit 10)
    logits <- mk_logits(c(20, 10, -1e4, -1e4))
    tok <- sample_tok(logits$clone(), ids1, 1.0, 1.0, 0.0, 10.0)
    expect_equal(as.integer(tok$cpu()), 2L)

    # Top-p keeps the token that crosses the threshold (HF behavior).
    # probs (0.5, 0.3, 0.2), top_p = 0.6: cumsum crosses at token 2, so
    # tokens 1-2 stay and token 3 is dropped.
    logits <- mk_logits(log(c(0.5, 0.3, 0.2)))
    seen <- integer(0)
    for (i in 1:200) {
        tok <- sample_tok(logits$clone(), no_ids, 1.0, 0.6, 0.0, 1.0)
        seen <- c(seen, as.integer(tok$cpu()))
    }
    expect_true(2L %in% seen)   # crossing token kept
    expect_false(3L %in% seen)  # beyond-threshold token dropped

    # top_p = 1.0 disables nucleus filtering entirely (Python default):
    # the tail token remains sampleable
    logits <- mk_logits(log(c(0.4, 0.35, 0.25)))
    seen <- integer(0)
    for (i in 1:300) {
        tok <- sample_tok(logits$clone(), no_ids, 1.0, 1.0, 0.0, 1.0)
        seen <- c(seen, as.integer(tok$cpu()))
    }
    expect_true(all(c(1L, 2L, 3L) %in% seen))

    # min-p: tokens below min_p * max_prob are unreachable
    logits <- mk_logits(log(c(0.90, 0.08, 0.02)))
    seen <- integer(0)
    for (i in 1:300) {
        tok <- sample_tok(logits$clone(), no_ids, 1.0, 1.0, 0.05, 1.0)
        seen <- c(seen, as.integer(tok$cpu()))
    }
    expect_false(3L %in% seen)  # 0.02 < 0.05 * 0.90
}
