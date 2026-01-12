# Voice Encoder for chatteRbox
# LSTM-based speaker embedding model

# ============================================================================
# Configuration
# ============================================================================

#' Voice encoder configuration
#'
#' @return List with configuration parameters
voice_encoder_config <- function() {
  list(
    num_mels = 40,
    sample_rate = 16000,
    speaker_embed_size = 256,
    ve_hidden_size = 256,
    n_fft = 400,
    hop_size = 160,
    win_size = 400,
    fmax = 8000,
    fmin = 0,
    ve_partial_frames = 160,
    ve_final_relu = TRUE,
    normalized_mels = FALSE
  )
}

# ============================================================================
# Mel Spectrogram for Voice Encoder
# ============================================================================

#' Compute mel spectrogram for voice encoder
#'
#' @param wav Audio samples (numeric vector)
#' @param config Voice encoder config
#' @return Mel spectrogram (time, n_mels)
compute_ve_mel <- function(wav, config = voice_encoder_config()) {
  # Convert to torch tensor if needed
  if (!inherits(wav, "torch_tensor")) {
    wav <- torch::torch_tensor(wav, dtype = torch::torch_float32())
  }

  # Use the S3Gen mel computation with VE parameters
  mel <- compute_mel_spectrogram(
    wav,
    n_fft = config$n_fft,
    n_mels = config$num_mels,
    sr = config$sample_rate,
    hop_size = config$hop_size,
    win_size = config$win_size,
    fmin = config$fmin,
    fmax = config$fmax,
    center = TRUE
  )

  # Transpose to (batch, time, mels)
  mel$transpose(2, 3)
}

# ============================================================================
# Voice Encoder Module
# ============================================================================

#' Voice encoder module
#'
#' @param config Voice encoder configuration
#' @return nn_module
voice_encoder <- torch::nn_module(
  "VoiceEncoder",

  initialize = function(config = NULL) {
    if (is.null(config)) {
      config <- voice_encoder_config()
    }
    self$config <- config

    # 3-layer LSTM
    self$lstm <- torch::nn_lstm(
      input_size = config$num_mels,
      hidden_size = config$ve_hidden_size,
      num_layers = 3,
      batch_first = TRUE
    )

    # Projection to speaker embedding
    self$proj <- torch::nn_linear(config$ve_hidden_size, config$speaker_embed_size)

    # Cosine similarity scaling parameters
    self$similarity_weight <- torch::nn_parameter(torch::torch_tensor(10.0))
    self$similarity_bias <- torch::nn_parameter(torch::torch_tensor(-5.0))
  },

  forward = function(mels) {
    # mels: (B, T, M) where T is partial_frames

    # Pass through LSTM
    lstm_out <- self$lstm(mels)
    hidden <- lstm_out[[2]][[1]]  # hidden states from all layers

    # Get final layer hidden state
    final_hidden <- hidden[3, , ]  # (B, hidden_size)

    # Project to embedding
    raw_embeds <- self$proj(final_hidden)

    # Apply ReLU if configured
    if (self$config$ve_final_relu) {
      raw_embeds <- torch::nnf_relu(raw_embeds)
    }

    # L2 normalize
    raw_embeds / torch::torch_norm(raw_embeds, dim = 2, keepdim = TRUE)
  },

  # Inference for full utterance with overlapping partials
  inference = function(mels, overlap = 0.5, min_coverage = 0.8) {
    config <- self$config
    device <- next(self$parameters())$device

    # Ensure mels is on device
    if (mels$device$type != device$type) {
      mels <- mels$to(device = device)
    }

    batch_size <- mels$size(1)
    n_frames <- mels$size(2)

    # Compute frame step based on overlap
    frame_step <- as.integer(round(config$ve_partial_frames * (1 - overlap)))

    # Compute number of partials
    n_partials <- (n_frames - config$ve_partial_frames + frame_step) %/% frame_step
    if (n_partials == 0) {
      n_partials <- 1
    }

    # Collect all partials
    all_partials <- list()
    for (b in seq_len(batch_size)) {
      for (i in seq_len(n_partials)) {
        start_idx <- (i - 1) * frame_step + 1
        end_idx <- min(start_idx + config$ve_partial_frames - 1, n_frames)

        # Handle short partials
        if (end_idx - start_idx + 1 < config$ve_partial_frames) {
          partial <- torch::torch_zeros(c(1, config$ve_partial_frames, config$num_mels),
                                        device = device)
          actual_len <- end_idx - start_idx + 1
          partial[1, 1:actual_len, ] <- mels[b, start_idx:end_idx, ]
        } else {
          partial <- mels[b, start_idx:end_idx, ]$unsqueeze(1)
        }
        all_partials[[length(all_partials) + 1]] <- partial
      }
    }

    # Stack and forward through network
    partials_tensor <- torch::torch_cat(all_partials, dim = 1)
    partial_embeds <- self(partials_tensor)

    # Average partial embeddings per utterance
    embeds <- list()
    idx <- 1
    for (b in seq_len(batch_size)) {
      batch_embeds <- partial_embeds[idx:(idx + n_partials - 1), ]
      mean_embed <- torch::torch_mean(batch_embeds, dim = 1, keepdim = TRUE)
      # L2 normalize
      mean_embed <- mean_embed / torch::torch_norm(mean_embed, dim = 2, keepdim = TRUE)
      embeds[[b]] <- mean_embed
      idx <- idx + n_partials
    }

    torch::torch_cat(embeds, dim = 1)
  }
)

# ============================================================================
# High-level Functions
# ============================================================================

#' Compute speaker embedding from audio
#'
#' @param model Voice encoder model
#' @param audio Audio samples (numeric vector or tensor)
#' @param sr Sample rate of audio
#' @param overlap Overlap between partials (default 0.5)
#' @return Speaker embedding tensor (1, 256)
#' @export
compute_speaker_embedding <- function(model, audio, sr, overlap = 0.5) {
  config <- model$config
  device <- next(model$parameters())$device

  # Convert to numeric if tensor
  if (inherits(audio, "torch_tensor")) {
    audio <- as.numeric(audio$cpu())
  }

  # Resample to 16kHz if needed
  if (sr != config$sample_rate) {
    audio <- resample_audio(audio, sr, config$sample_rate)
  }

  # Compute mel spectrogram
  mel <- compute_ve_mel(audio, config)
  mel <- mel$to(device = device)

  # Run inference
  torch::with_no_grad({
    embed <- model$inference(mel, overlap = overlap)
  })

  embed
}

#' Load voice encoder weights from safetensors
#'
#' @param model Voice encoder model
#' @param state_dict Named list of tensors
#' @return Model with loaded weights
load_voice_encoder_weights <- function(model, state_dict) {
  torch::with_no_grad({
    # LSTM weights
    # PyTorch uses 0-indexed layer names (weight_ih_l0), R torch uses 1-indexed (weight_ih_l1)
    for (py_layer in 0:2) {
      r_layer <- py_layer + 1  # Convert to R torch 1-indexed

      # Keys in the safetensors file (PyTorch naming)
      weight_ih_key <- paste0("lstm.weight_ih_l", py_layer)
      weight_hh_key <- paste0("lstm.weight_hh_l", py_layer)
      bias_ih_key <- paste0("lstm.bias_ih_l", py_layer)
      bias_hh_key <- paste0("lstm.bias_hh_l", py_layer)

      # R torch parameter names (1-indexed)
      r_weight_ih <- paste0("weight_ih_l", r_layer)
      r_weight_hh <- paste0("weight_hh_l", r_layer)
      r_bias_ih <- paste0("bias_ih_l", r_layer)
      r_bias_hh <- paste0("bias_hh_l", r_layer)

      if (weight_ih_key %in% names(state_dict)) {
        model$lstm$parameters[[r_weight_ih]]$copy_(state_dict[[weight_ih_key]])
      }
      if (weight_hh_key %in% names(state_dict)) {
        model$lstm$parameters[[r_weight_hh]]$copy_(state_dict[[weight_hh_key]])
      }
      if (bias_ih_key %in% names(state_dict)) {
        model$lstm$parameters[[r_bias_ih]]$copy_(state_dict[[bias_ih_key]])
      }
      if (bias_hh_key %in% names(state_dict)) {
        model$lstm$parameters[[r_bias_hh]]$copy_(state_dict[[bias_hh_key]])
      }
    }

    # Projection layer
    if ("proj.weight" %in% names(state_dict)) {
      model$proj$weight$copy_(state_dict[["proj.weight"]])
    }
    if ("proj.bias" %in% names(state_dict)) {
      model$proj$bias$copy_(state_dict[["proj.bias"]])
    }

    # Similarity parameters
    if ("similarity_weight" %in% names(state_dict)) {
      model$similarity_weight$copy_(state_dict[["similarity_weight"]])
    }
    if ("similarity_bias" %in% names(state_dict)) {
      model$similarity_bias$copy_(state_dict[["similarity_bias"]])
    }
  })

  model
}
