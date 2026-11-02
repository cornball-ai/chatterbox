# The built-in voice ships with the package and loads as a voice_embedding
# of the shapes each model set expects. Needs torch/lantern, not weights.

for (name in c("chatterbox", "chatterbox-turbo")) {
    expect_true(nzchar(system.file("voices", paste0(name, ".safetensors"),
                                   package = "chatterbox")))
}

if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    std <- chatterbox::builtin_voice()
    expect_inherits(std, "voice_embedding")
    expect_equal(dim(std$ve_embedding), c(1, 256))
    expect_equal(dim(std$cond_prompt_speech_tokens), c(1, 150))
    expect_equal(dim(std$ref_dict$prompt_token), c(1, 157))
    expect_equal(as.numeric(std$ref_dict$prompt_token_len), 157)
    expect_equal(dim(std$ref_dict$prompt_feat), c(1, 314, 80))
    expect_equal(dim(std$ref_dict$embedding), c(1, 192))
    expect_null(std$ref_dict$prompt_feat_len)
    expect_identical(as.character(std$cond_prompt_speech_tokens$dtype), "Long")
    expect_identical(as.character(std$ref_dict$prompt_feat$dtype), "Float")
    expect_true(is.na(std$sample_rate))

    turbo <- chatterbox::builtin_voice(turbo = TRUE)
    expect_equal(dim(turbo$cond_prompt_speech_tokens), c(1, 375))
    expect_equal(dim(turbo$ref_dict$prompt_token), c(1, 250))
    expect_equal(dim(turbo$ref_dict$prompt_feat), c(1, 500, 80))

    # loaded once per model set and device
    expect_identical(chatterbox::builtin_voice(), std)

    # a model object picks its own variant and device
    m <- chatterbox::chatterbox(device = "cpu", turbo = TRUE, load = FALSE)
    expect_identical(chatterbox::builtin_voice(m), turbo)

    # what generate() does with its voice argument
    expect_identical(chatterbox:::.resolve_voice(m, NULL), turbo)
    expect_identical(chatterbox:::.resolve_voice(m, std), std)
    expect_error(chatterbox:::.resolve_voice(m, 42), "built-in voice")

    expect_stdout(print(std), "built-in voice")
}
