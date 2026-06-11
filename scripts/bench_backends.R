# Benchmark the three T3 inference backends on current torch.
# Question: does the C++ backend (src/) still earn its keep vs pure R
# and JIT-traced on torch 0.17 / libtorch 2.8?

library(chatterbox)

# Short enough (~130 speech tokens) to fit the traced/cpp 350-position
# cache with room to spare, so all backends finish with natural EOS and
# do identical work
text <- "Hello world, this is a test of the chatterbox text to speech system"
ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")

model <- chatterbox(device = "cuda")
model <- load_chatterbox(model)
voice <- create_voice_embedding(model, ref_audio)

run <- function (label, ...) {
    t0 <- Sys.time()
    res <- generate(model, text, voice, ...)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("%-12s %6.1fs total  %4d tokens  %6.0f ms/token  eos=%s  %.2fs audio\n",
        label, secs, res$n_tokens, 1000 * secs / max(res$n_tokens, 1),
        res$eos_found, res$audio_sec))
}

cat("torch:", as.character(packageVersion("torch")), "\n\n")

run("pure-R")
run("cpp", backend = "cpp")
run("traced-cold", traced = TRUE)
run("traced-warm", traced = TRUE)
run("cpp-warm", backend = "cpp")
