# Validate R ports of torchaudio resample + kaldi fbank against the
# Python reference (chatterbox-tts:blackwell container).
#
# Workflow:
#   1. cd /home/troy/chatterbox && r scripts/test_kaldi_resample.R
#      (writes /tmp/test_signal.csv; exits if reference files missing)
#   2. docker run --rm -v /tmp:/tmp -v /home/troy/chatterbox/scripts:/scripts \
#        chatterbox-tts:blackwell python /scripts/save_kaldi_resample_ref.py
#   3. cd /home/troy/chatterbox && r scripts/test_kaldi_resample.R
#      (compares R ports against the references)
#
# Acceptance: max abs diff < 1e-4 (resample), < 1e-3 (fbank, log domain).

library(torch)

source("R/resample.R")
source("R/kaldi_fbank.R")

# --- 1. Deterministic test signal: 2.5 s @ 24000 Hz -------------------------

set.seed(1)
sr <- 24000
n <- as.integer(2.5 * sr)
tt <- (0:(n - 1)) / sr
signal <- 0.5 * sin(2 * pi * 440 * tt) +
    0.25 * sin(2 * pi * 1357 * tt) +
    0.15 * sin(2 * pi * 5000 * tt) +
    0.05 * rnorm(n)

writeLines(sprintf("%.17g", signal), "/tmp/test_signal.csv")
cat(sprintf("signal: %d samples @ %d Hz -> /tmp/test_signal.csv\n", n, sr))

if (!file.exists("/tmp/ref_resample.csv") || !file.exists("/tmp/ref_fbank.csv")) {
    cat("Reference files missing. Generate them with:\n")
    cat("  docker run --rm -v /tmp:/tmp -v /home/troy/chatterbox/scripts:/scripts \\\n")
    cat("    chatterbox-tts:blackwell python /scripts/save_kaldi_resample_ref.py\n")
    quit(status = 1)
}

# --- 2. Resample 24000 -> 16000 ----------------------------------------------

ref_resample <- as.numeric(readLines("/tmp/ref_resample.csv"))
r_resample <- sinc_resample(signal, 24000, 16000)

cat(sprintf("resample: R length %d, ref length %d\n",
    length(r_resample), length(ref_resample)))
stopifnot(length(r_resample) == length(ref_resample))

diff_resample <- max(abs(r_resample - ref_resample))
cat(sprintf("resample max abs diff: %.3g\n", diff_resample))

# Type round-trip: tensor in -> tensor out
r_resample_t <- sinc_resample(
    torch_tensor(signal, dtype = torch_float32()), 24000, 16000
)
stopifnot(inherits(r_resample_t, "torch_tensor"), r_resample_t$dim() == 1)

# --- 3. Kaldi fbank of the resampled signal ----------------------------------

ref_fbank_flat <- as.numeric(readLines("/tmp/ref_fbank.csv"))
n_frames_ref <- length(ref_fbank_flat) / 80
cat(sprintf("ref fbank: %d frames x 80 bins\n", n_frames_ref))

# Feed the reference resampled signal so fbank is tested in isolation
r_fbank <- kaldi_fbank(ref_resample, num_mel_bins = 80L, sample_rate = 16000)
cat(sprintf("R fbank: [%s]\n", paste(dim(r_fbank), collapse = ", ")))
stopifnot(dim(r_fbank)[1] == n_frames_ref, dim(r_fbank)[2] == 80)

# Row-major flatten matches Python's .flatten()
r_fbank_flat <- as.numeric(r_fbank$reshape(- 1))
diff_fbank <- max(abs(r_fbank_flat - ref_fbank_flat))
cat(sprintf("fbank max abs diff: %.3g\n", diff_fbank))

# End-to-end: R resample feeding R fbank vs Python reference chain
r_fbank_e2e <- kaldi_fbank(r_resample, num_mel_bins = 80L, sample_rate = 16000)
diff_e2e <- max(abs(as.numeric(r_fbank_e2e$reshape(- 1)) - ref_fbank_flat))
cat(sprintf("fbank end-to-end (R resample -> R fbank) max abs diff: %.3g\n",
    diff_e2e))

# Orientation check for the CAMPPlus caller: feature - feature.mean(dim=0)
ref_mat <- matrix(ref_fbank_flat, ncol = 80, byrow = TRUE)
ref_centered <- sweep(ref_mat, 2, colMeans(ref_mat))
r_centered <- r_fbank - r_fbank$mean(dim = 1, keepdim = TRUE)
diff_centered <- max(abs(as.numeric(r_centered$reshape(- 1)) -
    as.numeric(t(ref_centered))))
cat(sprintf("fbank mean-centered (CAMPPlus path) max abs diff: %.3g\n",
    diff_centered))

# --- 4. Verdict ---------------------------------------------------------------

ok_resample <- diff_resample < 1e-4
ok_fbank <- diff_fbank < 1e-3
cat(sprintf("resample %s (< 1e-4), fbank %s (< 1e-3)\n",
    if (ok_resample) "PASS" else "FAIL",
    if (ok_fbank) "PASS" else "FAIL"))
if (!ok_resample || !ok_fbank) {
    quit(status = 1)
}
