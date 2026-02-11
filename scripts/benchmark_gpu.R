#!/usr/bin/env r
# GPU Benchmark: Rtorch vs Python container
#
# Compares CUDA inference across:
#   1. Rtorch (native R + libtorch, current rtorch-port branch)
#   2. Python container (chatterbox-tts:blackwell via HTTP API)

library(chatterbox)

text <- "The quick brown fox jumps over the lazy dog."
ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")
n_warm <- 5

cat("============================================================\n")
cat("           GPU BENCHMARK: Rtorch vs Python\n")
cat("============================================================\n")
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))
cat(sprintf("Text: \"%s\"\n", text))
cat(sprintf("Warm runs: %d\n\n", n_warm))

# ============================================================================
# 1. Rtorch backend (CUDA)
# ============================================================================

cat("=== Loading Rtorch model ===\n")
t_load <- system.time({
    model <- chatterbox("cuda")
    model <- load_chatterbox(model)
})
cat(sprintf("Model load: %.1fs\n", t_load[3]))

voice <- create_voice_embedding(model, ref_audio)
gc()

cat("\n=== Rtorch (CUDA, float32) ===\n")

# Cold
gc()
t0 <- proc.time()
result <- generate(model, text, voice)
cold <- (proc.time() - t0)[3]
cold_dur <- length(result$audio) / result$sample_rate
cat(sprintf("  Cold:   %6.2fs  (%.1fs audio)\n", cold, cold_dur))
rm(result); gc()

# Warm
rtorch_times <- numeric(n_warm)
rtorch_durs <- numeric(n_warm)
for (i in seq_len(n_warm)) {
    gc()
    t0 <- proc.time()
    result <- generate(model, text, voice)
    rtorch_times[i] <- (proc.time() - t0)[3]
    rtorch_durs[i] <- length(result$audio) / result$sample_rate
    cat(sprintf("  Warm %d: %6.2fs  (%.1fs audio)\n", i, rtorch_times[i], rtorch_durs[i]))
    rm(result)
}
gc()

rtorch_warm <- mean(rtorch_times)
rtorch_dur <- mean(rtorch_durs)

# VRAM (allocated by libtorch caching allocator, not nvidia-smi total)
vram_rtorch <- Rtorch::cuda_memory_stats()["allocated_current"] / 1024^2

# Unload
rm(model, voice); gc()
Sys.sleep(2)

# ============================================================================
# 2. Python container (HTTP API, CUDA, float32)
# ============================================================================

cat("\n=== Python container ===\n")
health <- tryCatch(
    system2("curl", c("-s", "http://localhost:7810/health"), stdout = TRUE, stderr = FALSE),
    error = function(e) NULL
)

if (is.null(health) || !grepl("healthy", paste(health, collapse = ""))) {
    cat("Container not available, skipping.\n")
    py_cold <- NA; py_warm <- NA; py_dur <- NA
} else {
    cat("Container healthy.\n")

    tmp_wav <- tempfile(fileext = ".wav")
    body_json <- sprintf(
        '{"input":"%s","voice":"default","exaggeration":0.5,"cfg_weight":0.5,"temperature":0.8}',
        text
    )
    tmp_body <- tempfile(fileext = ".json")
    writeLines(body_json, tmp_body)

    call_container <- function() {
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
        dur
    }

    # Cold
    t0 <- proc.time()
    py_cold_dur <- call_container()
    py_cold <- (proc.time() - t0)[3]
    cat(sprintf("  Cold:   %6.2fs  (%.1fs audio)\n", py_cold, py_cold_dur))

    # Warm
    py_times <- numeric(n_warm)
    py_durs_vec <- numeric(n_warm)
    for (i in seq_len(n_warm)) {
        t0 <- proc.time()
        py_durs_vec[i] <- call_container()
        py_times[i] <- (proc.time() - t0)[3]
        cat(sprintf("  Warm %d: %6.2fs  (%.1fs audio)\n", i, py_times[i], py_durs_vec[i]))
    }

    py_warm <- mean(py_times)
    py_dur <- mean(py_durs_vec)

    unlink(tmp_body)
    unlink(tmp_wav)
}

# VRAM for container
vram_py <- as.numeric(
    jsonlite::fromJSON(paste(health, collapse = ""))$memory_info$gpu_memory_allocated_mb
)

# ============================================================================
# Results
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("                      RESULTS\n")
cat("============================================================\n")
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))
cat(sprintf("Precision: float32 (all backends)\n\n"))

cat(sprintf("%-20s  %10s  %10s  %10s  %10s  %10s\n",
    "Backend", "Cold Start", "Warm Mean", "Audio", "RT Factor", "VRAM"))
cat(sprintf("%-20s  %10s  %10s  %10s  %10s  %10s\n",
    "-------", "----------", "---------", "-----", "---------", "----"))

# Rtorch
rtf_rtorch <- rtorch_dur / rtorch_warm
cat(sprintf("%-20s  %9.2fs  %9.2fs  %9.1fs  %9.2fx  %7.0f MB\n",
    "Rtorch (R+libtorch)", cold, rtorch_warm, rtorch_dur, rtf_rtorch, vram_rtorch))

# Python
if (!is.na(py_warm)) {
    rtf_py <- py_dur / py_warm
    cat(sprintf("%-20s  %9.2fs  %9.2fs  %9.1fs  %9.2fx  %7.0f MB\n",
        "Python (container)", py_cold, py_warm, py_dur, rtf_py, vram_py))
}

cat(sprintf("\nSpeedup: Python is %.1fx faster than Rtorch (warm start)\n",
    rtorch_warm / py_warm))
cat(sprintf("Per-token: Rtorch ~%.0fms, Python ~%.0fms\n",
    rtorch_warm / (rtorch_dur * 25 / 4) * 1000,  # ~25 tokens/s, 4s audio
    py_warm / (py_dur * 25 / 4) * 1000))
