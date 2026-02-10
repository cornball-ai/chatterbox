#!/usr/bin/env r
# Benchmark: R backend vs C++ backend vs container
#
# Compares wall-clock time for generating speech across three backends:
#   1. chatterbox R backend (backend = "r")
#   2. chatterbox C++ backend (backend = "cpp")
#   3. chatterbox-tts container (HTTP API on port 7810)
#
# Runs native benchmarks first, then frees VRAM before container test
# to avoid OOM on 16GB GPUs.

library(chatterbox)
library(torch)

# ============================================================================
# Setup
# ============================================================================

device <- if (cuda_is_available()) "cuda" else "cpu"
cat(sprintf("Device: %s\n", device))
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))

text <- "The quick brown fox jumps over the lazy dog."
ref_audio <- system.file("audio", "jfk.mp3", package = "chatterbox")
n_runs <- 3

cat(sprintf("\nText: \"%s\"\n", text))
cat(sprintf("Reference: %s\n", ref_audio))
cat(sprintf("Runs per backend: %d\n\n", n_runs))

# ============================================================================
# Benchmark helper
# ============================================================================

benchmark <- function(label, fn, n = 3) {
    times <- numeric(n)
    durations <- numeric(n)
    for (i in seq_len(n)) {
        gc(); tryCatch(cuda_empty_cache(), error = function(e) NULL)
        t <- system.time(result <- fn())
        times[i] <- t[3]
        if (is.list(result) && "audio" %in% names(result)) {
            durations[i] <- length(result$audio) / result$sample_rate
        }
        cat(sprintf("  Run %d: %.2fs (%.1fs audio)\n", i, times[i], durations[i]))
    }
    list(
        label = label,
        times = times,
        mean_time = mean(times),
        mean_duration = mean(durations)
    )
}

# ============================================================================
# Phase 1: Native R and C++ backends
# ============================================================================

cat("=== Loading native model ===\n")
t_load <- system.time({
    model <- chatterbox(device)
    model <- load_chatterbox(model)
})
cat(sprintf("Model load time: %.1fs\n", t_load[3]))

cat("Computing voice embedding...\n")
voice <- create_voice_embedding(model, ref_audio)

# Warmup
cat("\nWarming up R backend...\n")
generate(model, "Hello", voice, backend = "r")
gc(); tryCatch(cuda_empty_cache(), error = function(e) NULL)

cat("Warming up C++ backend...\n")
generate(model, "Hello", voice, backend = "cpp")
gc(); tryCatch(cuda_empty_cache(), error = function(e) NULL)

cat(sprintf("\nVRAM after warmup: %s\n",
    system("nvidia-smi --query-gpu=memory.used --format=csv,noheader", intern = TRUE)))

# Benchmark native backends
cat("\n=== R Backend ===\n")
r_result <- benchmark("R", function() {
    generate(model, text, voice, backend = "r")
}, n_runs)

cat("\n=== C++ Backend ===\n")
cpp_result <- benchmark("C++", function() {
    generate(model, text, voice, backend = "cpp")
}, n_runs)

# ============================================================================
# Phase 2: Free native model, run container benchmark
# ============================================================================

cat("\n=== Unloading native model ===\n")
rm(model, voice)
gc(); tryCatch(cuda_empty_cache(), error = function(e) NULL)
cat(sprintf("VRAM after unload: %s\n",
    system("nvidia-smi --query-gpu=memory.used --format=csv,noheader", intern = TRUE)))

# Check container health
cat("\n=== Container (HTTP API) ===\n")
health <- tryCatch(
    system2("curl", c("-s", "http://localhost:7810/health"), stdout = TRUE, stderr = FALSE),
    error = function(e) NULL
)
if (is.null(health) || !grepl("healthy", paste(health, collapse = ""))) {
    cat("Container not available, skipping.\n")
    container_result <- list(
        label = "Container", times = rep(NA, n_runs),
        mean_time = NA, mean_duration = NA
    )
} else {
    cat("Container healthy.\n")

    # Warmup
    cat("Warming up container...\n")
    system2("curl", c("-s", "-X", "POST", "http://localhost:7810/v1/audio/speech",
        "-H", "Content-Type: application/json",
        "-d", '{"input":"Hello","voice":"default"}',
        "-o", "/dev/null"), stdout = FALSE, stderr = FALSE)

    container_times <- numeric(n_runs)
    container_durations <- numeric(n_runs)
    tmp_wav <- tempfile(fileext = ".wav")

    # Write JSON body to temp file to avoid shell quoting issues
    tmp_body <- tempfile(fileext = ".json")
    writeLines(sprintf(
        '{"input":"%s","voice":"default","exaggeration":0.5,"cfg_weight":0.5,"temperature":0.8}',
        text
    ), tmp_body)

    for (i in seq_len(n_runs)) {
        cmd <- sprintf(
            'curl -s -X POST http://localhost:7810/v1/audio/speech -H "Content-Type: application/json" -d @%s -o %s',
            tmp_body, tmp_wav
        )
        t <- system.time(system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE))
        container_times[i] <- t[3]

        if (file.exists(tmp_wav) && file.size(tmp_wav) > 100) {
            wav <- tryCatch(tuneR::readWave(tmp_wav), error = function(e) NULL)
            if (!is.null(wav)) {
                container_durations[i] <- length(wav@left) / wav@samp.rate
            }
        }
        cat(sprintf("  Run %d: %.2fs (%.1fs audio)\n", i, container_times[i], container_durations[i]))
    }
    unlink(tmp_body)
    unlink(tmp_wav)

    container_result <- list(
        label = "Container",
        times = container_times,
        mean_time = mean(container_times),
        mean_duration = mean(container_durations)
    )
}

# ============================================================================
# Summary
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("                    BENCHMARK RESULTS\n")
cat("============================================================\n")
cat(sprintf("Text: \"%s\"\n", text))
cat(sprintf("GPU: %s\n\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))

results <- list(r_result, cpp_result, container_result)

cat(sprintf("%-12s  %8s  %8s  %8s\n", "Backend", "Time", "Audio", "RT Factor"))
cat(sprintf("%-12s  %8s  %8s  %8s\n", "-------", "----", "-----", "---------"))
for (r in results) {
    if (is.na(r$mean_time)) {
        cat(sprintf("%-12s  %8s  %8s  %8s\n", r$label, "N/A", "N/A", "N/A"))
    } else {
        rtf <- r$mean_duration / r$mean_time
        cat(sprintf("%-12s  %7.2fs  %7.1fs  %7.2fx\n",
                    r$label, r$mean_time, r$mean_duration, rtf))
    }
}

if (!is.na(container_result$mean_time)) {
    cat(sprintf("\nC++ vs R speedup: %.1fx\n", r_result$mean_time / cpp_result$mean_time))
    cat(sprintf("Container vs R speedup: %.1fx\n", r_result$mean_time / container_result$mean_time))
    cat(sprintf("Container vs C++ speedup: %.1fx\n", cpp_result$mean_time / container_result$mean_time))
} else {
    cat(sprintf("\nC++ vs R speedup: %.1fx\n", r_result$mean_time / cpp_result$mean_time))
}
