# End-to-end GPU validation of the python-parity-fixes branch.
# Runs chris-english's failing inputs (issues #1, #2) plus controls,
# reports eos_found / token counts / audio stats, writes wavs to ~/Sync.

library(chatterbox)

out_dir <- path.expand("~/Sync/chatterbox-parity")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")

model <- chatterbox(device = "cuda")
model <- load_chatterbox(model)

cases <- list(
    list(label = "control_hello",
         text = "Hello world, this is a test of the chatterbox text to speech system"),
    list(label = "issue2_mixedcase",
         text = "Yes, Rarely or never Almost never, at most once in a while, over the past week Sometimes Only a couple of days over the past week, not many times in any given day Often Four or more days over the past week, several times each day Very Often just about every day over the past week, multiple times throughout the Day."),
    list(label = "issue1_emphasis",
         text = "Very often. As I said earlier, homework is a daily battle. He will do anything to get out of it. My head hurts; My stomach hurts. He asks for snacks or stops working to get a glass of water."),
    list(label = "issue1_long",
         text = "This would be moderate problem because both his teacher and I spend the time to organize Harry and make sure he has taken home his assignments and returned them to school, but no plan is foolproof."),
    list(label = "paraling_laughter",
         text = "Well that is just wonderful news. [laughter] I could not be happier for you.")
)

voice <- create_voice_embedding(model, ref_audio)

results <- data.frame()
for (cs in cases) {
    cat("\n=== ", cs$label, " ===\n", sep = "")
    t0 <- Sys.time()
    res <- tryCatch(
        generate(model, cs$text, voice),
        error = function (e) {
            cat("ERROR:", conditionMessage(e), "\n")
            NULL
        }
    )
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (is.null(res)) {
        results <- rbind(results, data.frame(
            label = cs$label, eos = NA, n_tokens = NA,
            audio_sec = NA, audio_std = NA, gen_sec = round(elapsed, 1)
        ))
        next
    }
    wav_path <- file.path(out_dir, paste0(cs$label, ".wav"))
    write_audio(res$audio, res$sample_rate, wav_path)
    results <- rbind(results, data.frame(
        label = cs$label,
        eos = res$eos_found,
        n_tokens = res$n_tokens,
        audio_sec = round(res$audio_sec, 2),
        audio_std = round(stats::sd(res$audio), 4),
        gen_sec = round(elapsed, 1)
    ))
}

cat("\n\n==== SUMMARY ====\n")
print(results, row.names = FALSE)
cat("\nWavs in:", out_dir, "\n")
