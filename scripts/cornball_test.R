#!/usr/bin/env Rscript
# TTS test: "cornball AI is doing something for our country!"

library(chatterbox)

cat("Loading model...\n")
model <- chatterbox("cpu")
model <- load_chatterbox(model)

text <- "cornball AI is doing something for our country!"
cat(sprintf("\nGenerating: '%s'\n", text))

result <- generate(model, text, "inst/audio/jfk.wav")

cat(sprintf("\nOutput: %d samples (%.2f sec)\n",
        length(result$audio), length(result$audio) / result$sample_rate))
cat(sprintf("Audio stats: mean=%.6f, std=%.6f\n",
        mean(result$audio), sd(result$audio)))

write_audio(result$audio, result$sample_rate, "outputs/cornball_jfk.wav")
cat("Saved to outputs/cornball_jfk.wav\n")

