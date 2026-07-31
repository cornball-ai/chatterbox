#!/usr/bin/env r
# Benchmark repeated activation of a resident chatterbox model.
#
# Usage: r tools/bench_resident.R [variant] [n]
#        Rscript --vanilla tools/bench_resident.R [variant] [n]
#   variant: "standard" (default) or "turbo"
#
# Compares the path residency replaces -- a cold chatterbox() load
# (disk -> CPU -> GPU) -- against resident_load() once followed by n
# activate/deactivate cycles (pinned RAM <-> GPU). GB/s is computed from
# the manifest's logical bytes; allocator numbers are printed as evidence
# only. Allocator GC options are applied by the loaders themselves.

suppressMessages({
    library(chatterbox)
    library(torch)
})

if (!exists("argv")) argv <- commandArgs(trailingOnly = TRUE)
variant <- if (length(argv) >= 1) argv[1] else "standard"
n <- if (length(argv) >= 2) as.integer(argv[2]) else 10L
turbo <- identical(variant, "turbo")

stopifnot(cuda_is_available())
alloc <- function() cuda_memory_stats()$allocated_bytes$all$current
gb <- function(b) b / 1024^3

cat("== bench_resident:", variant, "x", n, "==\n")
smi <- tryCatch(system2("nvidia-smi",
    c("--query-gpu=name,memory.total", "--format=csv,noheader"),
    stdout = TRUE)[1], error = function(e) "nvidia-smi unavailable")
cat("gpu:", smi, "\n")
cat(sprintf("allocator baseline: %.3f GB\n", gb(alloc())))

# --- the path residency replaces ---
t_cold <- system.time(
    m <- chatterbox("cuda", turbo = turbo))[["elapsed"]]
peak_cold <- alloc()
cat(sprintf("cold chatterbox() load:  %8.2f s   (allocator %.2f GB)\n",
    t_cold, gb(peak_cold)))
rm(m)
invisible(gc())
cuda_empty_cache()

# --- resident path ---
t_load <- system.time(
    res <- resident_load(turbo = turbo, verbose = FALSE))[["elapsed"]]
s <- resident_status(res)
bytes <- s$pinned_bytes
cat(sprintf("resident_load():         %8.2f s   (pins %.2f GB, %d artifacts, rev %s)\n",
    t_load, gb(bytes), length(s$identity$artifacts),
    substr(s$identity$revision, 1, 12)))

act <- numeric(n)
deact <- numeric(n)
peak_active <- 0
for (i in seq_len(n)) {
    act[i] <- system.time(resident_activate(res))[["elapsed"]]
    peak_active <- max(peak_active, alloc())
    deact[i] <- system.time(resident_deactivate(res))[["elapsed"]]
}
after_evict <- alloc()

cat(sprintf("activate   x%d: median %.3f s  min %.3f  max %.3f  (%.2f GB/s)\n",
    n, median(act), min(act), max(act), gb(bytes) / median(act)))
cat(sprintf("deactivate x%d: median %.3f s  min %.3f  max %.3f\n",
    n, median(deact), min(deact), max(deact)))
cat(sprintf("allocator: active %.2f GB, after evict %.3f GB\n",
    gb(peak_active), gb(after_evict)))
cat(sprintf("speedup vs cold load: %.0fx (%.2f s -> %.3f s)\n",
    t_cold / median(act), t_cold, median(act)))

# sanity: the reactivated model still speaks
resident_activate(res)
voice <- system.file("audio", "jfk.wav", package = "chatterbox")
r <- resident_generate(res, "Resident model sanity check.", voice,
    top_k = 1L, max_new_tokens = 120L, backend = "jit")
cat("generate sanity:", length(r$audio), "samples at", r$sample_rate, "Hz\n")
resident_deactivate(res)
resident_unload(res)
