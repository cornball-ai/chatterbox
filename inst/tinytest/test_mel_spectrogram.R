library(chatterbox)

# compute_mel_spectrogram on a synthetic sine wave (CPU torch)
if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    sr <- 24000
    duration <- 0.5
    t <- seq(0, duration, length.out = sr * duration)
    y <- sin(2 * pi * 440 * t)

    spec <- compute_mel_spectrogram(y, sr = sr)
    expect_inherits(spec, "torch_tensor")
    # Expect (batch, n_mels, time)
    expect_equal(spec$dim(), 3L)
    spec_shape <- as.integer(spec$shape)
    expect_equal(spec_shape[1], 1L)
    expect_equal(spec_shape[2], 80L)
    expect_true(spec_shape[3] > 0L)
}
