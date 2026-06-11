library(chatterbox)

# Sentence-initial caps preserved; mid-sentence caps lowercased
expect_equal(
    normalize_tts_text("Yes, Rarely or never Almost never."),
    "Yes, rarely or never almost never."
)

# Pronoun "I" stays capitalized mid-sentence
expect_equal(
    normalize_tts_text("As I said earlier, homework is a battle."),
    "As I said earlier, homework is a battle."
)

# Caps after sentence boundary stay capitalized
expect_equal(
    normalize_tts_text("Very often. As I said earlier."),
    "Very often. As I said earlier."
)

# Caps after semicolon get lowercased (mid-sentence)
expect_equal(
    normalize_tts_text("My head hurts; My stomach hurts."),
    "My head hurts; my stomach hurts."
)

# All-caps emphasis words get lowercased
expect_equal(
    normalize_tts_text("This is ALERT level."),
    "This is alert level."
)

# Internal caps (camelCase / weirdCase) get lowercased
expect_equal(
    normalize_tts_text("The rarelY pattern."),
    "The rarely pattern."
)

# Empty / non-character inputs pass through
expect_equal(normalize_tts_text(""), "")
expect_equal(normalize_tts_text(NA_character_), NA_character_)
