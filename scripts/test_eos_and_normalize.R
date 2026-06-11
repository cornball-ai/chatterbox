# Manual GPU smoke test for PR #4: EOS detection + text normalization
# Reproduces the failing inputs from chris-english issues #1 and #2.

library(chatterbox)

ref_audio <- "inst/audio/jfk.wav"
out_dir <- tempfile("chatterbox_test_")
dir.create(out_dir)

model <- chatterbox(device = "cuda")
model <- load_chatterbox(model)

# Issue #2: mixed-case input
text_mixed <- "Yes, Rarely or never Almost never, at most once in a while, over the past week Sometimes Only a couple of days over the past week, not many times in any given day Often Four or more days over the past week, several times each day Very Often just about every day over the past week, multiple times throughout the Day."

# Issue #1: long input that previously hit token cap
text_long <- "Very often. As I said earlier, homework is a daily battle. He will do anything to get out of it. My head hurts; My stomach hurts. He asks for snacks or stops working to get a glass of water."

# Issue #1: another reported failure
text_long2 <- "This would be moderate problem because both his teacher and I spend the time to organize Harry and make sure he has taken home his assignments and returned them to school, but no plan is foolproof."

cases <- list(
    list(label = "issue#2 mixed-case",       text = text_mixed),
    list(label = "issue#1 emphasis",         text = text_long),
    list(label = "issue#1 long sentence",    text = text_long2)
)

results <- list()

for (i in seq_along(cases)) {
    case <- cases[[i]]
    cat("\n========================================\n")
    cat("Case ", i, ": ", case$label, "\n", sep = "")
    cat("Input: ", substr(case$text, 1, 80), "...\n", sep = "")
    cat("Normalized: ", substr(normalize_tts_text(case$text), 1, 80), "...\n", sep = "")

    out_path <- file.path(out_dir, sprintf("case_%d.wav", i))
    res <- tts_to_file(model, case$text, ref_audio, out_path)

    cat("\nResult:\n")
    cat("  path       : ", res$path, "\n")
    cat("  eos_found  : ", res$eos_found, "\n")
    cat("  n_tokens   : ", res$n_tokens, "\n")
    cat("  audio_sec  : ", round(res$audio_sec, 2), "\n")

    results[[i]] <- data.frame(
        case = case$label,
        path = res$path,
        eos_found = res$eos_found,
        n_tokens = res$n_tokens,
        audio_sec = round(res$audio_sec, 2),
        stringsAsFactors = FALSE
    )
}

cat("\n========================================\n")
cat("Summary report:\n")
print(do.call(rbind, results))

cat("\nWAV files written to: ", out_dir, "\n", sep = "")
