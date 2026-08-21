# The split that was actually spoken, carried out of tts_chunked.
#
# `heardThrough`-style reporting -- a player saying how many chunks reached
# the speaker before it was cut -- is only meaningful because chunk i says
# text piece i. If the piece is recovered afterwards by re-splitting the
# source, any drift (a changed separator, different whitespace, slightly
# different model output) moves the boundaries, and the transcript is cut
# in the wrong place while still reading as a plausible sentence. So the
# split leaves with the audio rather than being reconstructible from it.
#
# No GPU here: `generate` is stubbed, which is the only thing between this
# and the real serial path. Everything else -- the split, the loop, the
# emission, the return -- is the shipping code.

library(chatterbox)

fake <- function(turbo = FALSE) structure(list(loaded = TRUE, turbo = turbo),
                                          class = "chatterbox")

# A non-character voice, so create_voice_embedding is never reached. The
# stub ignores it; what matters is that it is not a path to resolve.
VOICE <- list(embedding = 1)

ns <- asNamespace("chatterbox")
with_generate <- function(stub, expr) {
    orig <- get("generate", envir = ns, inherits = FALSE)
    assignInNamespace("generate", stub, ns = "chatterbox")
    on.exit(assignInNamespace("generate", orig, ns = "chatterbox"), add = TRUE)
    force(expr)
}

# One sample of audio per chunk, valued by the chunk's length, so a
# mis-paired emission is visible in the audio as well as the text.
stub_gen <- function(model, text, voice, ...) {
    list(audio = as.numeric(nchar(text)), sample_rate = 24000L)
}

TEXT <- "First sentence here. Second one follows. And a third to finish."

# ---- the emitted text is the piece that chunk says ---------------------
local({
    seen <- list()
    res <- with_generate(stub_gen,
        suppressMessages(tts_chunked(fake(), TEXT, VOICE, chunk_size = 25L,
            strategy = "serial",
            on_chunk = function(audio, index, total, text) {
                seen[[length(seen) + 1L]] <<- list(i = index, n = total,
                                                   t = text, a = audio)
            })))

    expect_true(length(seen) > 1L)
    idx <- vapply(seen, function(s) s$i, integer(1L))
    txt <- vapply(seen, function(s) s$t, character(1L))

    # Emitted in order, one per chunk, and `total` is the chunk count.
    expect_equal(idx, seq_along(res$chunks))
    expect_true(all(vapply(seen, function(s) s$n, integer(1L)) ==
                    length(res$chunks)))

    # THE MAPPING. What arrived at the callback is exactly what came back
    # in the return -- so a caller that recorded pieces as they played and
    # a caller that read them at the end agree. Asserted against res$chunks
    # rather than against a hand-written expected split: a literal here
    # would only ever agree with itself, and would need editing every time
    # the splitter's boundaries move for an unrelated reason.
    expect_equal(txt, res$chunks)

    # And the audio was generated FROM that piece, not merely delivered
    # next to it -- an emission that paired chunk i's audio with piece j
    # passes a text-only check.
    expect_equal(vapply(seen, function(s) s$a, numeric(1L)),
                 as.numeric(nchar(res$chunks)))
})

# ---- the split comes back whether or not anyone was listening ----------
# A caller that synthesizes the whole utterance and reads the split at the
# end must get the same thing, or "what was spoken" depends on whether
# someone happened to pass a callback.
local({
    quiet <- with_generate(stub_gen,
        suppressMessages(tts_chunked(fake(), TEXT, VOICE, chunk_size = 25L,
                                     strategy = "serial")))
    loud <- with_generate(stub_gen,
        suppressMessages(tts_chunked(fake(), TEXT, VOICE, chunk_size = 25L,
            strategy = "serial", on_chunk = function(a, i, n, t) NULL)))
    expect_equal(quiet$chunks, loud$chunks)
    expect_equal(quiet$audio, loud$audio)
})

# ---- rejoining the pieces covers the input -----------------------------
# The split is a partition of what was said, not a summary of it. Without
# this, a splitter that dropped a clause would still satisfy every
# index-mapping assertion above.
local({
    res <- with_generate(stub_gen,
        suppressMessages(tts_chunked(fake(), TEXT, VOICE, chunk_size = 25L,
                                     strategy = "serial")))
    expect_equal(gsub("[ ]+", " ", paste(res$chunks, collapse = " ")),
                 gsub("[ ]+", " ", TEXT))
})

# ---- empty input still answers with a vector ---------------------------
# character(0), not NULL. One shape that is sometimes absent is how a
# length() check lands on the wrong branch. This path returns before the
# model is touched, so no stub is needed.
local({
    res <- tts_chunked(fake(), "   ", VOICE)
    expect_true("chunks" %in% names(res))
    expect_equal(res$chunks, character(0))
    expect_equal(length(res$audio), 0L)
})

# ---- the arity is named in the refusal ---------------------------------
# A caller passing a three-argument callback gets "unused argument" from R
# with no clue which argument or whose contract. The guard's message is
# the only place the shape is stated at the call site.
e <- tryCatch(tts_chunked(fake(), "hi", VOICE, on_chunk = 42),
              error = conditionMessage)
expect_true(grepl("text", e))
