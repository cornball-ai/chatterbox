#!/usr/bin/env r
# GPU Benchmark: Rtorch vs Rtorch+compile vs Python container
#
# Compares CUDA inference across:
#   1. Rtorch (native R + libtorch)
#   2. Rtorch + compile (fused kernels via torchlang)
#   3. Python container (chatterbox-tts:blackwell via HTTP API)

library(chatterbox)

text <- "The quick brown fox jumps over the lazy dog."
ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")
n_warm <- 5

cat("============================================================\n")
cat("     GPU BENCHMARK: Rtorch vs Compiled vs Python\n")
cat("============================================================\n")
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))
cat(sprintf("Text: \"%s\"\n", text))
cat(sprintf("Warm runs: %d\n\n", n_warm))

# Helper: benchmark a loaded model
bench_model <- function(model, voice, label) {
    cat(sprintf("\n=== %s ===\n", label))

    # Cold
    gc(); Rtorch::cuda_empty_cache()
    t0 <- proc.time()
    result <- generate(model, text, voice)
    cold <- (proc.time() - t0)[3]
    cold_dur <- length(result$audio) / result$sample_rate
    cat(sprintf("  Cold:   %6.2fs  (%.1fs audio)\n", cold, cold_dur))
    rm(result); gc(); Rtorch::cuda_empty_cache()

    vram <- Rtorch::cuda_memory_stats()["allocated_current"] / 1024^2

    # Warm (tryCatch each run to tolerate OOM)
    times <- numeric(0)
    durs <- numeric(0)
    for (i in seq_len(n_warm)) {
        gc(); Rtorch::cuda_empty_cache()
        res <- tryCatch({
            t0 <- proc.time()
            result <- generate(model, text, voice)
            elapsed <- (proc.time() - t0)[3]
            dur <- length(result$audio) / result$sample_rate
            rm(result)
            list(time = elapsed, dur = dur)
        }, error = function(e) {
            cat(sprintf("  Warm %d: FAILED (%s)\n", i, conditionMessage(e)))
            NULL
        })
        if (!is.null(res)) {
            times <- c(times, res$time)
            durs <- c(durs, res$dur)
            cat(sprintf("  Warm %d: %6.2fs  (%.1fs audio)\n", i, res$time, res$dur))
        }
    }
    gc(); Rtorch::cuda_empty_cache()

    if (length(times) == 0) {
        cat("  No successful warm runs\n")
        return(list(cold = cold, warm = NA, dur = NA, vram = vram))
    }
    cat(sprintf("  (%d/%d warm runs succeeded)\n", length(times), n_warm))
    list(cold = cold, warm = mean(times), dur = mean(durs), vram = vram)
}

# ============================================================================
# 1. Rtorch (no compilation)
# ============================================================================

cat("=== Loading Rtorch model ===\n")
t_load <- system.time({
    model <- chatterbox("cuda")
    model <- load_chatterbox(model)
})
cat(sprintf("Model load: %.1fs\n", t_load[3]))

voice <- create_voice_embedding(model, ref_audio)
gc()

res_plain <- bench_model(model, voice, "Rtorch (CUDA, float32)")

rm(model, voice); gc()
Rtorch::cuda_empty_cache()
Sys.sleep(2)

# ============================================================================
# 2. Rtorch + compile (fused kernels)
# ============================================================================

cat("\n=== Loading Rtorch model + compile ===\n")
t_compile <- system.time({
    model_c <- chatterbox("cuda")
    model_c <- load_chatterbox(model_c, compiled = TRUE)
})
cat(sprintf("Model load + compile: %.1fs\n", t_compile[3]))

voice_c <- create_voice_embedding(model_c, ref_audio)
gc()

res_compiled <- bench_model(model_c, voice_c, "Rtorch + compile (CUDA, float32)")

rm(model_c, voice_c); gc()
Rtorch::cuda_empty_cache()
Sys.sleep(2)

# ============================================================================
# 3. Python container (HTTP API, CUDA, float32)
# ============================================================================

cat("\n=== Python container ===\n")
health <- tryCatch(
    system2("curl", c("-s", "http://localhost:7810/health"), stdout = TRUE, stderr = FALSE),
    error = function(e) NULL
)

py_cold <- NA; py_warm <- NA; py_dur <- NA; vram_py <- NA

if (is.null(health) || !grepl("healthy", paste(health, collapse = ""))) {
    cat("Container not available, skipping.\n")
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

    vram_py <- as.numeric(
        jsonlite::fromJSON(paste(health, collapse = ""))$memory_info$gpu_memory_allocated_mb
    )

    unlink(tmp_body)
    unlink(tmp_wav)
}

# ============================================================================
# Results
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("                      RESULTS\n")
cat("============================================================\n")
cat(sprintf("GPU: %s\n", system("nvidia-smi --query-gpu=name --format=csv,noheader", intern = TRUE)))
cat(sprintf("Precision: float32 (all backends)\n\n"))

cat(sprintf("%-22s  %10s  %10s  %10s  %10s  %10s\n",
    "Backend", "Cold Start", "Warm Mean", "Audio", "RT Factor", "VRAM"))
cat(sprintf("%-22s  %10s  %10s  %10s  %10s  %10s\n",
    "-------", "----------", "---------", "-----", "---------", "----"))

row <- function(label, r) {
    if (is.na(r$warm)) {
        cat(sprintf("%-22s  %9.2fs  %10s  %10s  %10s  %7.0f MB\n",
            label, r$cold, "N/A", "N/A", "N/A", r$vram))
    } else {
        rtf <- r$dur / r$warm
        cat(sprintf("%-22s  %9.2fs  %9.2fs  %9.1fs  %9.2fx  %7.0f MB\n",
            label, r$cold, r$warm, r$dur, rtf, r$vram))
    }
}

row("Rtorch", res_plain)
row("Rtorch + compile", res_compiled)

if (!is.na(py_warm)) {
    rtf_py <- py_dur / py_warm
    cat(sprintf("%-22s  %9.2fs  %9.2fs  %9.1fs  %9.2fx  %7.0f MB\n",
        "Python (container)", py_cold, py_warm, py_dur, rtf_py, vram_py))
}

# Comparisons
if (!is.na(res_plain$warm) && !is.na(res_compiled$warm)) {
    cat(sprintf("\nCompiled vs plain: %.2fx speedup (warm start)\n",
        res_plain$warm / res_compiled$warm))
}
if (!is.na(py_warm) && !is.na(res_plain$warm)) {
    cat(sprintf("Python vs plain:   %.2fx speedup (warm start)\n",
        res_plain$warm / py_warm))
}
if (!is.na(py_warm) && !is.na(res_compiled$warm)) {
    cat(sprintf("Python vs compiled: %.2fx speedup (warm start)\n",
        res_compiled$warm / py_warm))
}
