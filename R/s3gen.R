# S3Gen - Speech Token to Waveform Generator
# Flow matching decoder + HiFiGAN vocoder

# Constants
S3GEN_SR <- 24000  # Output sample rate

# ============================================================================
# Utility Functions
# ============================================================================

#' Create padding mask
#'
#' @param lengths Sequence lengths
#' @param max_len Maximum length
#' @return Boolean mask (TRUE for padded positions)
make_pad_mask <- function(lengths, max_len = NULL) {
  if (is.null(max_len)) {
    max_len <- max(as.integer(lengths$cpu()))
  }

  batch_size <- lengths$size(1)
  device <- lengths$device

  # Create range tensor
  range_tensor <- torch::torch_arange(0, max_len, device = device)$unsqueeze(1)

  # Compare with lengths
  lengths <- lengths$view(c(1, batch_size))
  range_tensor >= lengths
}

# ============================================================================
# Conformer Encoder (Simplified)
# ============================================================================

#' Upsample Conformer Encoder
#'
#' @param input_size Input dimension
#' @param output_size Output dimension
#' @param num_blocks Number of conformer blocks
#' @return nn_module
upsample_conformer_encoder <- torch::nn_module(
  "UpsampleConformerEncoder",

  initialize = function(input_size = 512, output_size = 512, num_blocks = 6) {
    self$input_size <- input_size
    self$output_size_val <- output_size

    # Input projection
    self$input_proj <- torch::nn_linear(input_size, output_size)

    # Simplified conformer blocks (using transformer encoder)
    encoder_layer <- torch::nn_transformer_encoder_layer(
      d_model = output_size,
      nhead = 8,
      dim_feedforward = 2048,
      dropout = 0.1,
      batch_first = TRUE
    )
    self$encoder <- torch::nn_transformer_encoder(encoder_layer, num_layers = num_blocks)

    # Upsample 2x (token to mel frame rate)
    self$upsample <- torch::nn_conv_transpose1d(output_size, output_size, kernel_size = 4,
                                                 stride = 2, padding = 1)
  },

  output_size = function() {
    self$output_size_val
  },

  forward = function(x, x_len) {
    # x: (batch, time, features)
    x <- self$input_proj(x)

    # Transformer encoding
    x <- self$encoder(x)

    # Upsample: (batch, time, features) -> (batch, features, time) -> upsample -> back
    x <- x$transpose(2, 3)
    x <- self$upsample(x)
    x <- x$transpose(2, 3)

    # Update lengths (2x)
    new_len <- x_len * 2

    list(x, new_len)
  }
)

# ============================================================================
# Conditional Flow Matching Decoder
# ============================================================================

#' CFM Estimator (UNet1D style)
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @return nn_module
cfm_estimator <- torch::nn_module(
  "CFMEstimator",

  initialize = function(in_channels = 320, out_channels = 80) {
    # Time embedding
    self$time_emb <- torch::nn_sequential(
      torch::nn_linear(1, 256),
      torch::nn_silu(),
      torch::nn_linear(256, 256)
    )

    # Input projection
    self$input_proj <- torch::nn_conv1d(in_channels, 256, 1)

    # Residual blocks
    self$blocks <- torch::nn_module_list(lapply(1:12, function(i) {
      torch::nn_sequential(
        torch::nn_conv1d(256, 256, 3, padding = 1),
        torch::nn_group_norm(32, 256),
        torch::nn_silu(),
        torch::nn_conv1d(256, 256, 3, padding = 1),
        torch::nn_group_norm(32, 256),
        torch::nn_silu()
      )
    }))

    # Output projection
    self$output_proj <- torch::nn_conv1d(256, out_channels, 1)
  },

  forward = function(x, mask, mu, t, spks, cond) {
    # x: (batch, 80, time) - noisy sample
    # mu: (batch, 80, time) - encoder output
    # t: (batch,) - timestep
    # spks: (batch, 80) - speaker embedding
    # cond: (batch, 80, time) - conditioning

    batch_size <- x$size(1)
    seq_len <- x$size(3)

    # Expand speaker embedding to sequence length
    spks_expanded <- spks$unsqueeze(3)$expand(c(-1, -1, seq_len))

    # Concatenate inputs: x + mu + spks + cond
    h <- torch::torch_cat(list(x, mu, spks_expanded, cond), dim = 2)
    h <- self$input_proj(h)

    # Time embedding
    t_emb <- self$time_emb(t$view(c(-1, 1)))$unsqueeze(3)

    # Residual blocks with time conditioning
    for (block in self$blocks) {
      residual <- h
      h <- block(h)
      h <- h + t_emb  # Add time embedding
      h <- h + residual  # Residual connection
    }

    # Output
    self$output_proj(h) * mask
  }
)

#' Causal Conditional Flow Matching
#'
#' @param in_channels Input channels (x + mu + spks + cond)
#' @param out_channels Output channels (mel bins)
#' @param spk_emb_dim Speaker embedding dimension
#' @return nn_module
causal_cfm <- torch::nn_module(
  "CausalConditionalCFM",

  initialize = function(in_channels = 320, out_channels = 80, spk_emb_dim = 80) {
    self$sigma_min <- 1e-6
    self$t_scheduler <- "cosine"
    self$inference_cfg_rate <- 0.7

    # Estimator network
    self$estimator <- cfm_estimator(in_channels, out_channels)

    # Pre-computed noise for reproducibility
    self$rand_noise <- torch::nn_buffer(
      torch::torch_randn(c(1, 80, 50 * 300))
    )
  },

  forward = function(mu, mask, spks, cond, n_timesteps = 10, temperature = 1.0) {
    device <- mu$device
    seq_len <- mu$size(3)

    # Initial noise
    z <- self$rand_noise[, , 1:seq_len]$to(device = device)$to(dtype = mu$dtype) * temperature

    # Time span with cosine schedule
    t_span <- torch::torch_linspace(0, 1, n_timesteps + 1, device = device, dtype = mu$dtype)
    if (self$t_scheduler == "cosine") {
      t_span <- 1 - torch::torch_cos(t_span * 0.5 * pi)
    }

    # Euler solver
    result <- self$solve_euler(z, t_span, mu, mask, spks, cond)

    list(result, NULL)  # Return mel and cache (NULL for now)
  },

  solve_euler = function(x, t_span, mu, mask, spks, cond) {
    batch_size <- x$size(1)
    seq_len <- x$size(3)
    device <- x$device

    t <- t_span[1]$unsqueeze(1)
    dt <- t_span[2] - t_span[1]

    # Pre-allocate tensors for CFG (batch size 2)
    x_in <- torch::torch_zeros(c(2, 80, seq_len), device = device, dtype = x$dtype)
    mask_in <- torch::torch_zeros(c(2, 1, seq_len), device = device, dtype = x$dtype)
    mu_in <- torch::torch_zeros(c(2, 80, seq_len), device = device, dtype = x$dtype)
    t_in <- torch::torch_zeros(2, device = device, dtype = x$dtype)
    spks_in <- torch::torch_zeros(c(2, 80), device = device, dtype = x$dtype)
    cond_in <- torch::torch_zeros(c(2, 80, seq_len), device = device, dtype = x$dtype)

    for (step in 2:length(t_span)) {
      # Classifier-Free Guidance: conditional and unconditional paths
      x_in[1:2, , ] <- x
      mask_in[1:2, , ] <- mask
      mu_in[1, , ] <- mu
      # mu_in[2] stays zero (unconditional)
      t_in[1:2] <- t
      spks_in[1, ] <- spks
      # spks_in[2] stays zero
      cond_in[1, , ] <- cond
      # cond_in[2] stays zero

      # Forward through estimator
      dphi_dt <- self$estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

      # CFG combination
      dphi_cond <- dphi_dt[1, , ]$unsqueeze(1)
      dphi_uncond <- dphi_dt[2, , ]$unsqueeze(1)
      dphi_dt <- (1.0 + self$inference_cfg_rate) * dphi_cond - self$inference_cfg_rate * dphi_uncond

      # Euler step
      x <- x + dt * dphi_dt
      t <- t + dt

      if (step < length(t_span)) {
        dt <- t_span[step + 1] - t_span[step]
      }
    }

    x$to(dtype = torch::torch_float32())
  }
)

# ============================================================================
# S3Gen Flow Module
# ============================================================================

#' Causal Masked Diff with Xvector
#'
#' @param vocab_size Speech token vocabulary size
#' @param input_size Token embedding size
#' @param output_size Mel bins
#' @param spk_embed_dim Speaker embedding dimension
#' @param input_frame_rate Input frame rate for audio processing
#' @param token_mel_ratio Ratio of tokens to mel frames
#' @return nn_module
causal_masked_diff_xvec <- torch::nn_module(
  "CausalMaskedDiffWithXvec",

  initialize = function(vocab_size = 6561, input_size = 512, output_size = 80,
                        spk_embed_dim = 192, input_frame_rate = 25, token_mel_ratio = 2) {
    self$vocab_size <- vocab_size
    self$input_size <- input_size
    self$output_size <- output_size
    self$input_frame_rate <- input_frame_rate
    self$token_mel_ratio <- token_mel_ratio
    self$pre_lookahead_len <- 3

    # Token embedding
    self$input_embedding <- torch::nn_embedding(vocab_size, input_size)

    # Speaker embedding projection
    self$spk_embed_affine_layer <- torch::nn_linear(spk_embed_dim, output_size)

    # Encoder
    self$encoder <- upsample_conformer_encoder(input_size, input_size, 6)

    # Encoder output projection
    self$encoder_proj <- torch::nn_linear(input_size, output_size)

    # Flow matching decoder
    self$decoder <- causal_cfm(in_channels = 320, out_channels = output_size, spk_emb_dim = output_size)
  },

  forward = function(token, token_len, prompt_token, prompt_token_len,
                     prompt_feat, prompt_feat_len, embedding, finalize = TRUE) {
    device <- token$device

    # Normalize and project speaker embedding
    embedding <- torch::nnf_normalize(embedding, dim = 2)
    embedding <- self$spk_embed_affine_layer(embedding)

    # Concatenate prompt and speech tokens
    token <- torch::torch_cat(list(prompt_token, token), dim = 2)
    token_len <- prompt_token_len + token_len

    # Create mask
    mask <- (!make_pad_mask(token_len))$unsqueeze(3)$to(dtype = embedding$dtype, device = device)

    # Clamp tokens to valid range
    token <- torch::torch_clamp(token, min = 0, max = self$vocab_size - 1)

    # Embed tokens
    token <- self$input_embedding(token + 1) * mask  # +1 for R indexing

    # Encode
    enc_result <- self$encoder(token, token_len)
    h <- enc_result[[1]]
    h_lengths <- enc_result[[2]]

    # Truncate lookahead if not finalizing
    if (!finalize) {
      h <- h[, 1:(h$size(2) - self$pre_lookahead_len * self$token_mel_ratio), ]
    }

    mel_len1 <- prompt_feat$size(2)
    mel_len2 <- h$size(2) - mel_len1

    # Project encoder output
    h <- self$encoder_proj(h)

    # Prepare conditioning
    conds <- torch::torch_zeros(c(1, mel_len1 + mel_len2, self$output_size),
                                device = device, dtype = h$dtype)
    conds[1, 1:mel_len1, ] <- prompt_feat
    conds <- conds$transpose(2, 3)

    # Create mask for decoder
    dec_mask <- torch::torch_ones(c(1, 1, mel_len1 + mel_len2), device = device, dtype = h$dtype)

    # Run decoder
    h <- h$transpose(2, 3)
    result <- self$decoder(
      mu = h,
      mask = dec_mask,
      spks = embedding$squeeze(2),
      cond = conds,
      n_timesteps = 10
    )
    feat <- result[[1]]

    # Extract generated portion (after prompt)
    feat <- feat[, , (mel_len1 + 1):(mel_len1 + mel_len2)]

    list(feat$to(dtype = torch::torch_float32()), NULL)
  }
)

# ============================================================================
# S3Gen Token2Wav (Full Pipeline)
# ============================================================================

#' S3Gen Token to Waveform
#'
#' @return nn_module
s3gen <- torch::nn_module(
  "S3Gen",

  initialize = function() {
    # Speech tokenizer for reference audio (128 mels for S3TokenizerV2)
    self$tokenizer <- s3_tokenizer()

    # Mel spectrogram extractor
    # (reuse from audio_utils)

    # Speaker encoder (CAMPPlus)
    self$speaker_encoder <- campplus()

    # Flow matching decoder
    self$flow <- causal_masked_diff_xvec()

    # HiFiGAN vocoder (will be added)
    self$mel2wav <- NULL

    # Fade-in to reduce artifacts
    n_trim <- S3GEN_SR %/% 50  # 20ms
    trim_fade <- torch::torch_zeros(2 * n_trim)
    fade_in <- (torch::torch_cos(torch::torch_linspace(pi, 0, n_trim)) + 1) / 2
    trim_fade[(n_trim + 1):(2 * n_trim)] <- fade_in
    self$trim_fade <- torch::nn_buffer(trim_fade)
  },

  #' Embed reference audio
  embed_ref = function(ref_wav, ref_sr, device = "auto") {
    if (device == "auto") {
      device <- self$tokenizer$mel_filters$device
    }

    # Convert to tensor
    if (!inherits(ref_wav, "torch_tensor")) {
      ref_wav <- torch::torch_tensor(ref_wav, dtype = torch::torch_float32())
    }

    if (ref_wav$dim() == 1) {
      ref_wav <- ref_wav$unsqueeze(1)
    }

    # Resample to 24kHz for mel extraction
    if (ref_sr != S3GEN_SR) {
      ref_wav_24 <- torch::torch_tensor(
        resample_audio(as.numeric(ref_wav$cpu()), ref_sr, S3GEN_SR),
        dtype = torch::torch_float32()
      )$unsqueeze(1)
    } else {
      ref_wav_24 <- ref_wav
    }

    # Compute mel spectrogram
    ref_mels <- compute_mel_spectrogram(ref_wav_24, sr = S3GEN_SR)
    ref_mels <- ref_mels$transpose(2, 3)$to(device = device)

    # Resample to 16kHz for speaker encoder and tokenizer
    ref_wav_16 <- torch::torch_tensor(
      resample_audio(as.numeric(ref_wav$cpu()), ref_sr, 16000),
      dtype = torch::torch_float32()
    )$unsqueeze(1)$to(device = device)

    # Speaker embedding (xvector)
    xvector <- compute_xvector_embedding(self$speaker_encoder, ref_wav_16, 16000)

    # Tokenize reference
    tok_result <- self$tokenizer(ref_wav_16)
    ref_tokens <- tok_result$tokens
    ref_token_lens <- tok_result$lens

    list(
      prompt_token = ref_tokens$to(device = device),
      prompt_token_len = ref_token_lens,
      prompt_feat = ref_mels,
      prompt_feat_len = NULL,
      embedding = xvector
    )
  },

  #' Run inference (tokens -> mel -> audio)
  inference = function(speech_tokens, ref_wav = NULL, ref_sr = NULL, ref_dict = NULL,
                       finalize = TRUE) {
    # Get reference dict
    if (is.null(ref_dict)) {
      if (is.null(ref_wav)) {
        stop("Must provide either ref_wav or ref_dict")
      }
      ref_dict <- self$embed_ref(ref_wav, ref_sr)
    }

    device <- ref_dict$embedding$device

    # Ensure tokens are 2D
    if (speech_tokens$dim() == 1) {
      speech_tokens <- speech_tokens$unsqueeze(1)
    }
    speech_tokens <- speech_tokens$to(device = device)
    speech_token_len <- torch::torch_tensor(speech_tokens$size(2), device = device)

    # Flow inference (tokens -> mel)
    result <- self$flow(
      token = speech_tokens,
      token_len = speech_token_len,
      prompt_token = ref_dict$prompt_token,
      prompt_token_len = ref_dict$prompt_token_len,
      prompt_feat = ref_dict$prompt_feat,
      prompt_feat_len = ref_dict$prompt_feat_len,
      embedding = ref_dict$embedding,
      finalize = finalize
    )
    output_mels <- result[[1]]

    # Vocoder (mel -> audio)
    if (!is.null(self$mel2wav)) {
      output_wavs <- self$mel2wav$inference(output_mels)

      # Apply fade-in
      fade_len <- length(self$trim_fade)
      output_wavs[, 1:fade_len] <- output_wavs[, 1:fade_len] * self$trim_fade

      return(list(output_wavs, NULL))
    }

    # Return mels if no vocoder
    list(output_mels, NULL)
  }
)

# ============================================================================
# Weight Loading
# ============================================================================

#' Load S3Gen weights
#'
#' @param model S3Gen model
#' @param state_dict State dictionary from safetensors
#' @return Model with loaded weights
#' @export
load_s3gen_weights <- function(model, state_dict) {
  # Helper to copy weight if exists
  copy_if_exists <- function(r_param, key) {
    if (key %in% names(state_dict)) {
      tryCatch({
        r_param$copy_(state_dict[[key]])
        return(TRUE)
      }, error = function(e) {
        warning("Failed to copy ", key, ": ", e$message)
        return(FALSE)
      })
    }
    FALSE
  }

  # ========== Speech Tokenizer ==========
  # Load tokenizer weights (S3TokenizerV2)
  load_s3tokenizer_weights(model$tokenizer, state_dict, prefix = "tokenizer.")

  # ========== Speaker Encoder (CAMPPlus) ==========
  load_campplus_weights(model$speaker_encoder, state_dict, prefix = "speaker_encoder.")

  # ========== Flow Module ==========
  # Input embedding
  copy_if_exists(model$flow$input_embedding$weight, "flow.input_embedding.weight")

  # Speaker embedding projection
  copy_if_exists(model$flow$spk_embed_affine_layer$weight, "flow.spk_embed_affine_layer.weight")
  copy_if_exists(model$flow$spk_embed_affine_layer$bias, "flow.spk_embed_affine_layer.bias")

  # Encoder projection
  copy_if_exists(model$flow$encoder_proj$weight, "flow.encoder_proj.weight")
  copy_if_exists(model$flow$encoder_proj$bias, "flow.encoder_proj.bias")

  # Encoder - this is a conformer with complex structure
  # For now, load what we can (the simplified transformer encoder)
  copy_if_exists(model$flow$encoder$input_proj$weight, "flow.encoder.embed.weight")
  copy_if_exists(model$flow$encoder$input_proj$bias, "flow.encoder.embed.bias")

  # Encoder upsample
  copy_if_exists(model$flow$encoder$upsample$weight, "flow.encoder.upsample.weight")
  copy_if_exists(model$flow$encoder$upsample$bias, "flow.encoder.upsample.bias")

  # CFM Decoder/Estimator - complex nested structure
  # The Python decoder is ConditionalDecoder (UNet-style)
  # We have simplified CFM estimator
  # Load what maps...

  # Time embedding
  copy_if_exists(model$flow$decoder$estimator$time_emb[[1]]$weight, "flow.decoder.estimator.time_embed.mlp.0.weight")
  copy_if_exists(model$flow$decoder$estimator$time_emb[[1]]$bias, "flow.decoder.estimator.time_embed.mlp.0.bias")
  copy_if_exists(model$flow$decoder$estimator$time_emb[[3]]$weight, "flow.decoder.estimator.time_embed.mlp.2.weight")
  copy_if_exists(model$flow$decoder$estimator$time_emb[[3]]$bias, "flow.decoder.estimator.time_embed.mlp.2.bias")

  # Input/output projections
  copy_if_exists(model$flow$decoder$estimator$input_proj$weight, "flow.decoder.estimator.input_projection.weight")
  copy_if_exists(model$flow$decoder$estimator$input_proj$bias, "flow.decoder.estimator.input_projection.bias")
  copy_if_exists(model$flow$decoder$estimator$output_proj$weight, "flow.decoder.estimator.output_projection.weight")
  copy_if_exists(model$flow$decoder$estimator$output_proj$bias, "flow.decoder.estimator.output_projection.bias")

  # ========== HiFiGAN Vocoder ==========
  if (!is.null(model$mel2wav)) {
    load_hifigan_weights(model$mel2wav, state_dict, prefix = "mel2wav.")
  }

  model
}

#' Load S3Gen from safetensors file
#'
#' @param path Path to s3gen.safetensors
#' @param device Device to load to ("cpu", "cuda", etc.)
#' @return S3Gen model with loaded weights
#' @export
load_s3gen <- function(path, device = "cpu") {
  # Read safetensors
  state_dict <- read_safetensors(path, device)

  # Create model
  model <- s3gen()

  # Create and attach vocoder
  model$mel2wav <- create_s3gen_vocoder(device)

  # Load weights
  model <- load_s3gen_weights(model, state_dict)

  # Move to device and eval mode
  model$to(device = device)
  model$eval()

  model
}
