# Turbo's GPT-2 tokenizer must emit paralinguistic/emotion tags ([sigh],
# [laugh], ...) as single special-token ids, not byte-BPE them into pieces.
# Tokenizer-only (no model), so it just needs the turbo files downloaded.
library(chatterbox)

# --- split_added_tokens(): core logic, always runs (no model files) ----------
sp <- chatterbox:::split_added_tokens

# Tag mid-text -> plain | tag(id) | plain
segs <- sp("hello [sigh] world", c("[sigh]" = 50268L))
expect_equal(length(segs), 3L)
expect_equal(segs[[1]]$text, "hello ")
expect_true(is.na(segs[[1]]$id))
expect_equal(segs[[2]]$text, "[sigh]")
expect_equal(segs[[2]]$id, 50268L)
expect_equal(segs[[3]]$text, " world")
expect_true(is.na(segs[[3]]$id))

# Tag only -> single special segment
only <- sp("[sigh]", c("[sigh]" = 50268L))
expect_equal(length(only), 1L)
expect_equal(only[[1]]$id, 50268L)

# No tag present -> single plain segment
none <- sp("just text", c("[sigh]" = 50268L))
expect_equal(length(none), 1L)
expect_true(is.na(none[[1]]$id))

# --- turbo GPT-2 tokenizer end-to-end (needs the turbo files) -----------------
if (turbo_models_available()) {
    tp <- chatterbox:::get_turbo_model_paths()
    tok <- chatterbox:::load_gpt2_tokenizer(tp$vocab, tp$merges, tp$added_tokens)

    expect_true(length(tok$added_tokens) >= 19L)

    # Each tag is one token with its declared id (from added_tokens.json).
    expect_equal(chatterbox:::tokenize_text_gpt2(tok, "[sigh]"), 50268L)
    expect_equal(chatterbox:::tokenize_text_gpt2(tok, "[laugh]"), 50275L)
    # A two-word tag still resolves to a single id.
    expect_equal(length(chatterbox:::tokenize_text_gpt2(tok, "[clear throat]")),
                 1L)

    # In running text the tag is one id and the words around it still BPE.
    ids <- chatterbox:::tokenize_text_gpt2(tok, "hello [sigh] world")
    expect_true(50268L %in% ids)
    expect_true(length(ids) > 1L)
}
