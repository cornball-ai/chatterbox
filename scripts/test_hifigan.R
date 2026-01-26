#!/usr/bin/env Rscript
# Test HiFiGAN vocoder against Python reference

library(chatterbox)

cat("=== HiFiGAN Vocoder Validation ===\n\n")

# Load Python reference
cat("Loading Python reference...\n")
ref <- chatterbox:::read_safetensors("outputs/hifigan_reference.safetensors")

cat(sprintf("Input mel: %s\n", paste(dim(ref$input_mel), collapse = "x")))
cat(sprintf("F0: %s\n", paste(dim(ref$f0), collapse = "x")))
cat(sprintf("Source: %s\n", paste(dim(ref$source), collapse = "x")))
cat(sprintf("Output audio: %s\n", paste(dim(ref$output_audio), collapse = "x")))

# Load weights
cat("\nLoading S3Gen weights...\n")
cache_dir <- chatterbox:::get_cache_dir()
weights_path <- file.path(cache_dir, "ResembleAI--chatterbox", "s3gen.safetensors")
state_dict <- chatterbox:::read_safetensors(weights_path)

# Create vocoder
cat("\nCreating HiFiGAN vocoder...\n")
vocoder <- chatterbox:::create_s3gen_vocoder("cpu")
vocoder$eval()

# Check architecture
cat(sprintf("  num_upsamples: %d\n", vocoder$num_upsamples))
cat(sprintf("  num_kernels: %d\n", vocoder$num_kernels))
cat(sprintf("  istft_n_fft: %d\n", vocoder$istft_n_fft))
cat(sprintf("  istft_hop_len: %d\n", vocoder$istft_hop_len))

# Check weight shapes
cat("\nWeight shapes:\n")
cat(sprintf("  conv_pre.weight: %s\n", paste(dim(vocoder$conv_pre$weight), collapse = "x")))
cat(sprintf("  conv_post.weight: %s\n", paste(dim(vocoder$conv_post$weight), collapse = "x")))
for (i in seq_along(vocoder$ups)) {
    cat(sprintf("  ups[%d].weight: %s\n", i, paste(dim(vocoder$ups[[i]]$weight), collapse = "x")))
}
for (i in seq_along(vocoder$source_downs)) {
    cat(sprintf("  source_downs[%d].weight: %s\n", i, paste(dim(vocoder$source_downs[[i]]$weight), collapse = "x")))
}

# Load weights
cat("\nLoading vocoder weights...\n")
vocoder <- chatterbox:::load_hifigan_weights(vocoder, state_dict, prefix = "mel2wav.")

# Test F0 predictor
cat("\n=== Test F0 Predictor ===\n")
torch::with_no_grad({
        input_mel <- ref$input_mel

        f0_r <- vocoder$f0_predictor$forward(input_mel)
        f0_py <- ref$f0

        cat(sprintf("R F0: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(f0_r), collapse = "x"), f0_r$mean()$item(), f0_r$std()$item()))
        cat(sprintf("Py F0: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(f0_py), collapse = "x"), f0_py$mean()$item(), f0_py$std()$item()))

        f0_diff <- (f0_r - f0_py)$abs()$max()$item()
        cat(sprintf("F0 max diff: %.6f\n", f0_diff))
    })

# Test F0 upsampling
cat("\n=== Test F0 Upsampling ===\n")
torch::with_no_grad({
        f0_up_r <- vocoder$f0_upsamp$forward(ref$f0$unsqueeze(2L))
        f0_up_py <- ref$f0_up

        cat(sprintf("R F0 up: shape=%s\n", paste(dim(f0_up_r), collapse = "x")))
        cat(sprintf("Py F0 up: shape=%s\n", paste(dim(f0_up_py), collapse = "x")))

        f0_up_diff <- (f0_up_r - f0_up_py)$abs()$max()$item()
        cat(sprintf("F0 up max diff: %.6f\n", f0_up_diff))
    })

# Test source module
cat("\n=== Test Source Module ===\n")
torch::with_no_grad({
        # F0 for source: (B, T, 1)
        f0_for_source <- ref$f0_up$transpose(2L, 3L)

        source_result <- vocoder$m_source$forward(f0_for_source)
        source_r <- source_result$sine_merge$transpose(2L, 3L)
        source_py <- ref$source

        cat(sprintf("R source: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(source_r), collapse = "x"), source_r$mean()$item(), source_r$std()$item()))
        cat(sprintf("Py source: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(source_py), collapse = "x"), source_py$mean()$item(), source_py$std()$item()))

        # Source generation is random, so we can't compare exactly
        # Just check that shapes match
        cat("(Source uses random noise, so exact match not expected)\n")
    })

# Test conv_pre
cat("\n=== Test conv_pre ===\n")
torch::with_no_grad({
        conv_pre_r <- vocoder$conv_pre$forward(ref$input_mel)
        conv_pre_py <- ref$conv_pre_out

        cat(sprintf("R conv_pre: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(conv_pre_r), collapse = "x"), conv_pre_r$mean()$item(), conv_pre_r$std()$item()))
        cat(sprintf("Py conv_pre: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(conv_pre_py), collapse = "x"), conv_pre_py$mean()$item(), conv_pre_py$std()$item()))

        conv_pre_diff <- (conv_pre_r - conv_pre_py)$abs()$max()$item()
        cat(sprintf("conv_pre max diff: %.6f\n", conv_pre_diff))
    })

# Test full inference with Python source
cat("\n=== Test Full Decode (with Python source) ===\n")
torch::with_no_grad({
        # Use Python's source to test decode path
        audio_r <- vocoder$decode(ref$input_mel, ref$source)
        audio_py <- ref$output_audio

        cat(sprintf("R audio: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(audio_r), collapse = "x"), audio_r$mean()$item(), audio_r$std()$item()))
        cat(sprintf("Py audio: shape=%s, mean=%.6f, std=%.6f\n",
                paste(dim(audio_py), collapse = "x"), audio_py$mean()$item(), audio_py$std()$item()))

        cat(sprintf("R audio range: [%.4f, %.4f]\n", audio_r$min()$item(), audio_r$max()$item()))
        cat(sprintf("Py audio range: [%.4f, %.4f]\n", audio_py$min()$item(), audio_py$max()$item()))

        audio_diff <- (audio_r - audio_py)$abs()$max()$item()
        cat(sprintf("\nMax diff: %.6f\n", audio_diff))

        if (audio_diff < 0.1) {
            cat("\n=== HIFIGAN VOCODER VALIDATED ===\n")
        } else {
            cat("\n=== NEEDS MORE WORK ===\n")

            # Debug: check intermediate shapes
            cat("\nDebugging decode path...\n")

            # STFT of source
            s_squeeze <- ref$source$squeeze(2L)
            cat(sprintf("  Source squeeze: %s\n", paste(dim(s_squeeze), collapse = "x")))
        }
    })

cat("\nDone.\n")

