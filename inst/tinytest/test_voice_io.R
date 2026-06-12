# Voice embedding save/load round-trip (no model weights needed, but
# torch/lantern must be installed - CI runners may lack it)

if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    fake_voice <- structure(
        list(
            ve_embedding = torch::torch_randn(1, 256),
            cond_prompt_speech_tokens = torch::torch_randint(0, 6561, c(1, 150),
                dtype = torch::torch_long()),
            ref_dict = list(
                prompt_token = torch::torch_randint(0, 6561, c(1, 250),
                    dtype = torch::torch_long()),
                prompt_token_len = torch::torch_tensor(250L),
                prompt_feat = torch::torch_randn(1, 500, 80),
                prompt_feat_len = NULL,
                embedding = torch::torch_randn(1, 192)
            ),
            sample_rate = 48000
        ),
        class = "voice_embedding"
    )

    path <- tempfile(fileext = ".voice")
    expect_identical(chatterbox::save_voice_embedding(fake_voice, path), path)
    expect_true(file.exists(path))

    loaded <- chatterbox::load_voice_embedding(path)
    expect_inherits(loaded, "voice_embedding")
    expect_identical(names(loaded), names(fake_voice))
    expect_identical(loaded$sample_rate, 48000)
    expect_null(loaded$ref_dict$prompt_feat_len)

    # tensor contents survive the round trip
    for (getter in list(
        function (v) v$ve_embedding,
        function (v) v$cond_prompt_speech_tokens,
        function (v) v$ref_dict$prompt_token,
        function (v) v$ref_dict$prompt_feat,
        function (v) v$ref_dict$embedding
    )) {
        expect_true(torch::torch_equal(getter(loaded), getter(fake_voice)))
    }

    # dtype preserved (long tokens stay long)
    expect_identical(
        as.character(loaded$ref_dict$prompt_token$dtype),
        as.character(fake_voice$ref_dict$prompt_token$dtype))

    expect_error(chatterbox::save_voice_embedding(list(), path),
        "voice_embedding")

    unlink(path)
}
