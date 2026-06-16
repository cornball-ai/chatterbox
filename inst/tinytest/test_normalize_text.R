library(chatterbox)

# --- normalize_internal_caps: the R-only caps mitigation (caps only) ----------
caps <- chatterbox:::normalize_internal_caps

# Sentence-initial caps preserved; mid-sentence caps lowercased
expect_equal(
    caps("Yes, Rarely or never Almost never."),
    "Yes, rarely or never almost never."
)

# Pronoun "I" stays capitalized mid-sentence
expect_equal(
    caps("As I said earlier, homework is a battle."),
    "As I said earlier, homework is a battle."
)

# Caps after sentence boundary stay capitalized
expect_equal(
    caps("Very often. As I said earlier."),
    "Very often. As I said earlier."
)

# Caps after semicolon get lowercased (mid-sentence)
expect_equal(
    caps("My head hurts; My stomach hurts."),
    "My head hurts; my stomach hurts."
)

# All-caps emphasis words get lowercased
expect_equal(caps("This is ALERT level."), "This is alert level.")

# Internal caps (camelCase / weirdCase) get lowercased
expect_equal(caps("The rarelY pattern."), "The rarely pattern.")

# Empty / non-character inputs pass through
expect_equal(caps(""), "")
expect_equal(caps(NA_character_), NA_character_)

# --- normalize_tts_text: the public wrapper (caps + punctuation) --------------

# Default: caps mitigation then punc_norm (note the appended period)
expect_equal(
    normalize_tts_text("This is ALERT level"),
    "This is alert level."
)

# caps = FALSE skips the mitigation; punc_norm still runs
expect_equal(
    normalize_tts_text("This is ALERT level", caps = FALSE),
    "This is ALERT level."
)

# Both off: text passes through untouched
expect_equal(
    normalize_tts_text("This is ALERT level", caps = FALSE,
                       punctuation = FALSE),
    "This is ALERT level"
)
