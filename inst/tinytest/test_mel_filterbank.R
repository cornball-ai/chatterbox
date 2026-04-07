library(chatterbox)

# Default Slaney filterbank: shape and basic invariants
fb <- chatterbox:::create_mel_filterbank(sr = 24000, n_fft = 1920, n_mels = 80)
expect_true(is.matrix(fb))
expect_equal(dim(fb), c(80L, 1920L %/% 2L + 1L))
expect_true(all(fb >= 0))
# Each filter should have at least one non-zero entry
expect_true(all(rowSums(fb) > 0))

# HTK formula path also works and returns same shape
fb_htk <- chatterbox:::create_mel_filterbank(sr = 24000, n_fft = 1920, n_mels = 80, htk = TRUE)
expect_equal(dim(fb_htk), dim(fb))
expect_true(all(fb_htk >= 0))

# fmax defaults to sr/2 -- explicit and implicit should match
fb_explicit <- chatterbox:::create_mel_filterbank(sr = 16000, n_fft = 512, n_mels = 40,
                                     fmax = 8000)
fb_implicit <- chatterbox:::create_mel_filterbank(sr = 16000, n_fft = 512, n_mels = 40)
expect_equal(fb_explicit, fb_implicit)
