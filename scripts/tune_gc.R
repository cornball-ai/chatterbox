# GC-knob tuning for the generation loop. One config per process
# (allocator settings are read once at torch startup). Usage:
#   r scripts/tune_gc.R <reserved_rate> <cpu_threshold_mb> <allocated_rate> <allocated_reserved_rate> [gens]

rate <- as.numeric(argv[1])
cpu_mb <- as.numeric(argv[2])
alloc_rate <- as.numeric(argv[3])
alloc_res_rate <- as.numeric(argv[4])
gens <- if (length(argv) >= 5) as.integer(argv[5]) else 3L

options(torch.cuda_allocator_reserved_rate = rate,
        torch.threshold_call_gc = cpu_mb,
        torch.cuda_allocator_allocated_rate = alloc_rate,
        torch.cuda_allocator_allocated_reserved_rate = alloc_res_rate)

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

for (i in seq_len(gens)) {
    t0 <- Sys.time()
    res <- generate(model, text, voice)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("cfg[res=%.2f cpu=%d alloc=%.2f allocres=%.2f] gen%d %5.0f ms/tok eos=%s vram=%dMB\n",
        rate, as.integer(cpu_mb), alloc_rate, alloc_res_rate, i,
        1000 * secs / max(res$n_tokens, 1), res$eos_found, vram_mb()))
}
