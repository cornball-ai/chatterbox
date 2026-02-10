#!/usr/bin/env r
# Full benchmark: cold start + warm start for R, C++, and container backends
#
# Measures:
#   - Cold start: first run after loading (includes tracing/compilation overhead)
#   - Warm start: mean of subsequent runs
#   - VRAM usage via cuda_memory_stats()

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
n_warm <- 3  # warm start runs

cat(sprintf("\nText: \"%s\"\n", text))
cat(sprintf("Reference: %s\n", ref_audio))
cat(sprintf("Warm runs per backend: %d\n", n_warm))

vram_mb <- function() {
    cuda_synchronize()
    stats <- cuda_memory_stats()
    stats$allocated_bytes$all$current / 1024^2
}

# ============================================================================
# Load model
# ============================================================================

cat("\n=== Loading model ===\n")
t_load <- system.time({
    model <- chatterbox(device)
    model <- load_chatterbox(model)
})
cat(sprintf("Model load time: %.1fs\n", t_load[3]))

voice <- create_voice_embedding(model, ref_audio)
gc(); cuda_empty_cache()
cat(sprintf("VRAM after load: %.0f MB allocated\n", vram_mb()))

# ============================================================================
# Benchmark function
# ============================================================================

run_benchmark <- function(label, fn, n_warm = 3) {
    cat(sprintf("\n=== %s ===\n", label))

    # Cold start
    gc(); cuda_empty_cache()
    t_cold <- system.time(result <- fn())
    cold_time <- t_cold[3]
    if (is.list(result) && "audio" %in% names(result)) {
        cold_dur <- length(result$audio) / result$sample_rate
    } else {
        cold_dur <- 0
    }
    cat(sprintf("  Cold:  %6.2fs  (%.1fs audio)\n", cold_time, cold_dur))
    rm(result); gc(); cuda_empty_cache()

    # Warm starts
    warm_times <- numeric(n_warm)
    warm_durs <- numeric(n_warm)
    for (i in seq_len(n_warm)) {
        gc(); cuda_empty_cache()
        t <- system.time(result <- fn())
        warm_times[i] <- t[3]
        if (is.list(result) && "audio" %in% names(result)) {
            warm_durs[i] <- length(result$audio) / result$sample_rate
        }
        cat(sprintf("  Warm %d: %6.2fs  (%.1fs audio)\n", i, warm_times[i], warm_durs[i]))
        rm(result)
    }
    gc(); cuda_empty_cache()

    list(
        label = label,
        cold_time = cold_time,
        cold_dur = cold_dur,
        warm_times = warm_times,
        warm_mean = mean(warm_times),
        warm_durs = warm_durs,
        warm_dur_mean = mean(warm_durs)
    )
}

# ============================================================================
# Native benchmarks
# ============================================================================

r_result <- run_benchmark("R backend", function() {
    generate(model, text, voice, backend = "r")
}, n_warm)

cpp_result <- run_benchmark("C++ backend", function() {
    generate(model, text, voice, backend = "cpp")
}, n_warm)

traced_result <- run_benchmark("R traced", function() {
    generate(model, text, voice, backend = "r", traced = TRUE)
}, n_warm)

# Peak VRAM during generation
gc(); cuda_empty_cache()
baseline_vram <- vram_mb()
dummy <- generate(model, text, voice, backend = "r")
peak_vram <- vram_mb()
rm(dummy); gc(); cuda_empty_cache()
steady_vram <- vram_mb()
cat(sprintf("\nVRAM: baseline=%.0f MB, peak=%.0f MB, steady=%.0f MB\n",
    baseline_vram, peak_vram, steady_vram))

# ============================================================================
# Unload native model, run container benchmark
# ============================================================================

cat("\n=== Unloading native model ===\n")
rm(model, voice)
gc(); cuda_empty_cache()
cat(sprintf("VRAM after unload: %.0f MB allocated\n", vram_mb()))

# Container benchmark
cat("\n=== Container (HTTP API) ===\n")
health <- tryCatch(
    system2("curl", c("-s", "http://localhost:7810/health"), stdout = TRUE, stderr = FALSE),
    error = function(e) NULL
)
if (is.null(health) || !grepl("healthy", paste(health, collapse = ""))) {
    cat("Container not available, skipping.\n")
    container_result <- list(
        label = "Container", cold_time = NA, cold_dur = NA,
        warm_mean = NA, warm_dur_mean = NA
    )
} else {
    cat("Container healthy.\n")

    tmp_wav <- tempfile(fileext = ".wav")
    tmp_body <- tempfile(fileext = ".json")
    writeLines(sprintf(
        '{"input":"%s","voice":"default","exaggeration":0.5,"cfg_weight":0.5,"temperature":0.8}',
        text
    ), tmp_body)

    container_fn <- function() {
        cmd <- sprintf(
            'curl -s -X POST http://localhost:7810/v1/audio/speech -H "Content-Type: application/json" -d @%s -o %s',
            tmp_body, tmp_wav
        )
        system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
        dur <- 0
        if (file.exists(tmp_wav) && file.size(tmp_wav) > 100) {
            wav <- tryCatch(tuneR::readWave(tmp_wav), error = function(e) NULL)
            if (!is.null(wav)) dur <- length(wav@left) / wav@samp.rate
        }
        list(audio = numeric(max(1, round(dur * 24000))), sample_rate = 24000)
    }

    container_result <- run_benchmark("Container", container_fn, n_warm)
    unlink(tmp_body)
    unlink(tmp_wav)
}

# ============================================================================
# Summary
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("                    BENCHMARK RESULTS\n")
cat("============================================================\n")
cat(sprintf("Text: \"%s\"\n", text))
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))
cat(sprintf("Precision: float32 (all backends)\n\n"))

results <- list(r_result, cpp_result, traced_result, container_result)

cat(sprintf("%-12s  %10s  %10s  %10s  %10s\n",
    "Backend", "Cold Start", "Warm Mean", "Audio", "RT Factor"))
cat(sprintf("%-12s  %10s  %10s  %10s  %10s\n",
    "-------", "----------", "---------", "-----", "---------"))
for (r in results) {
    if (is.na(r$warm_mean)) {
        cat(sprintf("%-12s  %10s  %10s  %10s  %10s\n",
            r$label, "N/A", "N/A", "N/A", "N/A"))
    } else {
        rtf <- r$warm_dur_mean / r$warm_mean
        cat(sprintf("%-12s  %9.2fs  %9.2fs  %9.1fs  %9.2fx\n",
            r$label, r$cold_time, r$warm_mean, r$warm_dur_mean, rtf))
    }
}

cat(sprintf("\nSpeedups (warm start, vs R backend):\n"))
cat(sprintf("  C++ backend: %.1fx\n", r_result$warm_mean / cpp_result$warm_mean))
cat(sprintf("  R traced:    %.1fx\n", r_result$warm_mean / traced_result$warm_mean))
if (!is.na(container_result$warm_mean)) {
    cat(sprintf("  Container:   %.1fx\n", r_result$warm_mean / container_result$warm_mean))
}

cat(sprintf("\nVRAM: model=%.0f MB, peak during generation=%.0f MB\n",
    baseline_vram, peak_vram))
