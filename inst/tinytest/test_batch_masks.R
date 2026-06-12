# Batched s3gen length/mask math (no weights needed)

if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    # make_pad_mask over a ragged batch
    lens <- torch::torch_tensor(c(3L, 5L, 1L))
    m <- chatterbox:::make_pad_mask(lens)
    expect_identical(dim(m), c(3L, 5L))
    expect_identical(as.logical(m[1, ]), c(FALSE, FALSE, FALSE, TRUE, TRUE))
    expect_identical(as.logical(m[3, ]), c(FALSE, TRUE, TRUE, TRUE, TRUE))

    # the (1,1) prompt_token_len broadcast bug: flattened sum keeps (B)
    prompt_len <- torch::torch_tensor(matrix(250L, 1, 1)) # ref_dict shape
    tok_len <- torch::torch_tensor(c(60L, 80L, 25L))
    total <- prompt_len$view(-1) + tok_len$view(-1)
    expect_identical(dim(total), 3L)
    expect_identical(as.integer(total), c(310L, 330L, 275L))
}

# generate_batch rejects unknown arguments instead of swallowing them
if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
    fake <- structure(list(loaded = TRUE, turbo = FALSE),
        class = "chatterbox")
    expect_error(
        chatterbox::generate_batch(fake, "hi", "v.wav", skip_vocoder = TRUE),
        "Unsupported arguments: skip_vocoder")
}
