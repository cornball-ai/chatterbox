# Turbo's GPT-2 tokenizer must emit paralinguistic/emotion tags ([sigh],
# [laugh], ...) as single special-token ids, not byte-BPE them into pieces.
# Tokenizer-only (no model), so it just needs the turbo files downloaded.
library(chatterbox)

if (turbo_models_available()) {
    tp <- get_turbo_model_paths()
    tok <- load_gpt2_tokenizer(tp$vocab, tp$merges, tp$added_tokens)

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
