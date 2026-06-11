# GC-knob tuning for the generation loop. One config per process
# (allocator state is sticky). Usage:
#   r scripts/tune_gc.R <reserved_rate> <cpu_threshold_mb> <gc_per_gen 0|1>

rate <- as.numeric(argv[1])
cpu_mb <- as.numeric(argv[2])
gc_per_gen <- identical(argv[3], "1")

options(torch.cuda_allocator_reserved_rate = rate,
        torch.threshold_call_gc = cpu_mb)

library(chatterbox)

text <- "Hello world, this is a test of the chatterbox text to speech system"
ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")
model <- chatterbox(device = "cuda")
model <- load_chatterbox(model)
voice <- create_voice_embedding(model, ref_audio)

vram_mb <- function () {
    as.integer(system2("nvidia-smi",
        c("--query-gpu=memory.used", "--format=csv,noheader,nounits"),
        stdout = TRUE)[1])
}

run <- function (label, ...) {
    t0 <- Sys.time()
    res <- generate(model, text, voice, ...)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (gc_per_gen) gc(full = TRUE)
    cat(sprintf("cfg[rate=%.2f cpu=%dMB gcgen=%d] %-11s %5.0f ms/tok  eos=%s  vram=%dMB\n",
        rate, as.integer(cpu_mb), as.integer(gc_per_gen), label,
        1000 * secs / max(res$n_tokens, 1), res$eos_found, vram_mb()))
}

run("pure-R-1")
run("pure-R-2")
invisible(generate(model, text, voice, traced = TRUE))
if (gc_per_gen) gc(full = TRUE)
run("traced-1", traced = TRUE)
run("traced-2", traced = TRUE)
run("traced-3", traced = TRUE)
