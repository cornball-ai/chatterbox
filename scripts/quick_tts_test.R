#!/usr/bin/env Rscript
# Quick TTS test after token indexing fix

library(chatterbox)

cat("Loading model...\n")
model <- chatterbox("cpu")
model <- load_chatterbox(model)

cat("\nGenerating speech...\n")
result <- generate(model, "Hello, this is a test", "inst/audio/jfk.wav")

cat(sprintf("\nOutput: %d samples (%.2f sec)\n",
        length(result$audio), length(result$audio) / result$sample_rate))

audio_vec <- result$audio
cat(sprintf("Audio stats: mean=%.6f, std=%.6f, range=[%.4f, %.4f]\n",
        mean(audio_vec), sd(audio_vec), min(audio_vec), max(audio_vec)))

if (sd(audio_vec) > 0.01) {
    cat("\nSUCCESS: Audio has signal\n")
    write_audio(result$audio, result$sample_rate, "outputs/quick_test.wav")
    cat("Saved to outputs/quick_test.wav\n")
} else {
    cat("\nFAILURE: Audio is still silent\n")
}

