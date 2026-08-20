# tts_chunked's strategy argument, at the point where it decides.
#
# No GPU: every assertion here is about a refusal or an argument check, all
# of which happen before the model is touched. What each strategy actually
# SYNTHESIZES is a live question and belongs to a live arm.
#
# The model stands in only for the two fields the guard reads -- `loaded`
# and `turbo`. That is the guard's whole contract, so a fixture carrying
# exactly those is testing the thing rather than a shape handed to it.

library(chatterbox)

fake <- function(turbo) structure(list(loaded = TRUE, turbo = turbo),
                                  class = "chatterbox")

# ---- batched on turbo is REFUSED, not silently downgraded ---------------
#
# The turbo weights have no batched implementation. Falling back to serial
# would hand the caller a different throughput profile than it asked for
# and say nothing -- and "asked for batched, got serial" is invisible from
# the return value, which is identical either way.
err <- tryCatch(tts_chunked(fake(TRUE), "hi", "v", strategy = "batched"),
                error = conditionMessage)
expect_true(is.character(err))
expect_true(grepl("turbo", err))
expect_true(grepl("no batched", err))
# It must name a way out, not just refuse.
expect_true(grepl("serial", err))

# ---- the other three combinations are NOT refused -----------------------
#
# Each must get past the strategy guard. They then fail later for want of a
# real model, which is the proof they were not stopped here: a guard that
# rejected everything would pass an "it errored" test.
past_guard <- function(...) {
    e <- tryCatch(tts_chunked(..., text = "hi", voice = "v"),
                  error = conditionMessage)
    !grepl("no batched", e)
}
expect_true(past_guard(fake(TRUE), strategy = "serial"))
expect_true(past_guard(fake(TRUE), strategy = "auto"))
expect_true(past_guard(fake(FALSE), strategy = "batched"))
expect_true(past_guard(fake(FALSE), strategy = "serial"))
expect_true(past_guard(fake(FALSE), strategy = "auto"))

# ---- an unknown strategy is refused by match.arg ------------------------
expect_error(tts_chunked(fake(FALSE), "hi", "v", strategy = "fastest"))

# ---- on_chunk must be a function ---------------------------------------
e <- tryCatch(tts_chunked(fake(FALSE), "hi", "v", on_chunk = 42),
              error = conditionMessage)
expect_true(grepl("on_chunk", e))

# ---- the default is unchanged behaviour --------------------------------
# Existing callers pass no strategy and must keep getting "auto". If this
# default ever moves, every current caller silently changes throughput
# profile.
expect_equal(eval(formals(tts_chunked)$strategy)[1], "auto")
