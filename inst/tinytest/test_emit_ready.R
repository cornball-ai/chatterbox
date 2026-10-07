# .emit_ready: the in-order flush behind tts_chunked's on_chunk callback.
#
# The batched (non-turbo) path synthesizes OUT OF ORDER -- it buckets chunks
# by speech-token length and walks `sort(unique(buckets))`, so chunk 5 can
# finish before chunk 1. Streaming that to a player in completion order would
# produce audio in the wrong order, which is worse than not streaming at all.
#
# So a chunk is emitted only once every chunk before it is done, and the
# watermark is what remembers how far that has got. Worst case (the last
# chunk completes last) this degenerates to emitting everything at the end,
# which is exactly today's behaviour.
#
# Each emission also carries the text piece that chunk says, so a caller
# recording what was spoken never has to re-derive the split.

er <- chatterbox:::.emit_ready

# A recorder standing in for the caller. It records rather than plays,
# because what is under test is WHICH chunks arrive and in WHAT ORDER.
rec <- function() {
    seen <- list()
    list(fn = function(audio, index, total, text) {
             seen[[length(seen) + 1L]] <<- list(a = audio, i = index,
                                                n = total, t = text)
             invisible(NULL)
         },
         idx = function() vapply(seen, function(s) s$i, integer(1L)),
         aud = function() vapply(seen, function(s) s$a, numeric(1L)),
         txt = function() vapply(seen, function(s) s$t, character(1L)),
         tot = function() vapply(seen, function(s) s$n, integer(1L)))
}

TX <- c("one", "two", "three")

# ---- everything ready: emits all, in order, watermark past the end ----
r <- rec()
w <- er(list(1, 2, 3), 1L, r$fn, 3L, TX)
expect_equal(r$idx(), 1:3)
expect_equal(r$aud(), c(1, 2, 3))
expect_equal(w, 4L)
# `total` is the chunk count, not the emitted count -- a caller sizing a
# progress bar off it would be wrong if this leaked the watermark instead.
expect_equal(r$tot(), rep(3L, 3))
# Chunk i says text piece i. Everything downstream that reports how far
# playback got means nothing without this mapping.
expect_equal(r$txt(), TX)

# ---- a HOLE stops the flush ----
# Position 2 is missing, so 3 must NOT be emitted even though it is ready.
# This is the assertion the whole helper exists for: without it the batched
# path streams chunk 3's audio before chunk 2's.
r <- rec()
w <- er(list(1, NULL, 3), 1L, r$fn, 3L, TX)
expect_equal(r$idx(), 1L)
expect_equal(w, 2L)

# ---- nothing ready: no calls, watermark unmoved ----
r <- rec()
w <- er(list(NULL, NULL), 1L, r$fn, 2L, TX[1:2])
expect_equal(length(r$idx()), 0L)
expect_equal(w, 1L)

# ---- the out-of-order fill, which is the real batched sequence ----
# Bucketing completes 3, then 1, then 2. Nothing may be emitted until 1
# lands, and when 2 lands it must carry 3 with it.
buf <- vector("list", 3L)
r <- rec()
w <- 1L

buf[[3]] <- 30                      # bucket with the long chunk finishes first
w <- er(buf, w, r$fn, 3L, TX)
expect_equal(length(r$idx()), 0L)   # 3 is ready and still must not go out
expect_equal(w, 1L)

buf[[1]] <- 10
w <- er(buf, w, r$fn, 3L, TX)
expect_equal(r$idx(), 1L)
expect_equal(w, 2L)

buf[[2]] <- 20
w <- er(buf, w, r$fn, 3L, TX)
expect_equal(r$idx(), c(1L, 2L, 3L))
expect_equal(r$aud(), c(10, 20, 30))
expect_equal(w, 4L)

# ---- ORDER IS THE CONTRACT, and this is the assertion that pins it ----
# Emitted order must equal original index order, not completion order. A
# test that only counted emissions would pass on a helper that streamed
# 3, 1, 2 -- which is the bug this replaces.
expect_true(!is.unsorted(r$idx()))

# TEXT FOLLOWS THE INDEX, NOT THE CALL. This flush resumed from watermark 2
# and emitted two chunks, so anything reading `texts` relative to where the
# flush started -- texts[[i - from + 1]], the natural-looking slip -- hands
# out "one", "two" for chunks 2 and 3. Both orders are three plausible
# sentences in the right sequence; only the absolute index is correct.
expect_equal(r$txt(), TX)

# ---- resuming from a watermark does not re-emit ----
# Called again with nothing new, it must stay silent rather than replay the
# prefix. A re-emitting flush would duplicate audio on every group boundary.
r2 <- rec()
w2 <- er(buf, w, r2$fn, 3L, TX)
expect_equal(length(r2$idx()), 0L)
expect_equal(w2, 4L)

# ---- a zero-length buffer is not an error ----
r <- rec()
expect_equal(er(list(), 1L, r$fn, 0L, character(0)), 1L)
expect_equal(length(r$idx()), 0L)

# ---- 1:1 is checked, not assumed ----
# A split that has drifted from the audio it labels is a programming error,
# and it stops here rather than emitting speech attributed to the wrong
# sentence. Checked before any emission: a partial flush followed by an
# error is the worst of both.
r <- rec()
expect_error(er(list(1, 2, 3), 1L, r$fn, 3L, TX[1:2]), "same length")
expect_equal(length(r$idx()), 0L)
expect_error(er(list(1, 2), 1L, r$fn, 2L, TX), "same length")
