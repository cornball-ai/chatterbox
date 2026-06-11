# Text tokenizer: punc_norm and BPE encoding
# Reference values verified against chatterbox-tts 0.1.4 (EnTokenizer)

# --- punc_norm (no model files needed) ---

pn <- chatterbox:::punc_norm

# Python: " ".join(text.split()) trims both ends and collapses runs
expect_equal(pn("  hello world"), "hello world.")
expect_equal(pn("multiple   spaces  here"), "Multiple spaces here.")
expect_equal(pn("tabs\tand\nnewlines"), "Tabs and newlines.")

# Leading space defeats capitalization in Python too (text[0] is " ")
expect_equal(pn(" lower start"), "lower start.")

# Trailing period appended only when missing
expect_equal(pn("Already ended."), "Already ended.")
expect_equal(pn("Question?"), "Question?")
expect_equal(pn("No ending"), "No ending.")

# Punctuation rewrites (Python leaves the double space these create;
# container-verified: punc_norm('Wait... what') == 'Wait,  what.')
expect_equal(pn("Wait... what"), "Wait,  what.")
expect_equal(pn("a: b; c"), "A, b,  c.")

# Empty input fallback
expect_equal(pn(""), "You need to add some text for me to talk.")

# --- BPE encoding (needs tokenizer.json from the HF cache) ---

if (at_home()) {
    tok_path <- list.files(
        path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots"),
        pattern = "^tokenizer\\.json$", recursive = TRUE, full.names = TRUE
    )
    if (length(tok_path) >= 1) {
        tok <- chatterbox:::load_tokenizer(tok_path[1])

        # 2-token full merge used to corrupt to c(40, UNK, dup)
        expect_equal(chatterbox:::tokenize_text(tok, "th"), 40L)

        # Added tokens are atomic, never spelled out
        expect_equal(chatterbox:::tokenize_text(tok, "[laughter]"), 607L)
        expect_equal(
            chatterbox:::tokenize_text(tok, "hi [sigh] ok"),
            c(21L, 22L, 2L, 611L, 2L, 166L)
        )

        # Spaces map 1:1 to [SPACE] (id 2)
        expect_equal(chatterbox:::tokenize_text(tok, "a  b"), c(14L, 2L, 2L, 15L))

        # Non-space whitespace is dropped but still separates words:
        # BPE must not merge across it ("ab" would be token 109)
        expect_equal(chatterbox:::tokenize_text(tok, "a\tb"), c(14L, 15L))
        expect_equal(chatterbox:::tokenize_text(tok, "a\n\nb"), c(14L, 15L))

        # Full-sentence regression, verified against the Python container
        expect_equal(
            chatterbox:::tokenize_text(
                tok, "Hello world, this is a test. [laughter] What about NASA and iPhone?"
            ),
            c(284L, 18L, 84L, 28L, 2L, 179L, 79L, 7L, 2L, 147L, 2L, 54L, 2L,
              14L, 2L, 33L, 218L, 9L, 2L, 607L, 2L, 299L, 21L, 48L, 2L, 215L,
              2L, 290L, 277L, 295L, 277L, 2L, 53L, 2L, 22L, 292L, 21L, 110L, 13L)
        )
    }
}
