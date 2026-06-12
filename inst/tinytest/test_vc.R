# voice_convert error paths (no weights needed)

if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    unloaded <- chatterbox::chatterbox("cpu")
    expect_error(chatterbox::voice_convert(unloaded, "x.wav", "y.wav"),
        "not loaded")

    fake_turbo <- structure(list(loaded = TRUE, turbo = TRUE),
        class = "chatterbox")
    expect_error(chatterbox::voice_convert(fake_turbo, "x.wav", "y.wav"),
        "non-turbo")

    fake <- structure(list(loaded = TRUE, turbo = FALSE),
        class = "chatterbox")
    fake_voice <- structure(list(ref_dict = list()),
        class = "voice_embedding")
    expect_error(chatterbox::voice_convert(fake, TRUE, fake_voice),
        "file path, numeric")
    expect_error(chatterbox::voice_convert(fake, c(0, 0.1), fake_voice),
        "sample_rate")
    expect_error(chatterbox::voice_convert(fake, c(0, 0.1), list()),
        "voice_embedding")
}
