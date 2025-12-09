# S3Tokenizer for chatteRbox
# Speech tokenizer that converts audio to discrete speech tokens

# Constants
S3_SR <- 16000  # Sample rate
S3_HOP <- 160   # STFT hop (100 frames/sec)
S3_TOKEN_RATE <- 25  # Tokens per second
SPEECH_VOCAB_SIZE <- 6561  # Vocabulary size

# ============================================================================
# Mel Spectrogram for S3Tokenizer
# ============================================================================

#' Compute log mel spectrogram for S3Tokenizer
#'
#' @param audio Audio tensor (batch, samples)
#' @param mel_filters Pre-computed mel filterbank
#' @param window Hann window
#' @param n_fft FFT size (default 400)
#' @param device Device
#' @return Log mel spectrogram (batch, n_mels, time)
s3_log_mel_spectrogram <- function(audio, mel_filters, window, n_fft = 400, device = "cpu") {
  if (!inherits(audio, "torch_tensor")) {
    audio <- torch::torch_tensor(audio, dtype = torch::torch_float32())
  }

  audio <- audio$to(device = device)
  window <- window$to(device = device)
  mel_filters <- mel_filters$to(device = device)

  # Compute STFT
  stft <- torch::torch_stft(
    audio,
    n_fft = n_fft,
    hop_length = S3_HOP,
    window = window,
    return_complex = TRUE
  )

  # Magnitude squared, drop last frame
  magnitudes <- torch::torch_abs(stft[, , 1:(stft$size(3) - 1)])$pow(2)

  # Apply mel filterbank
  mel_spec <- torch::torch_matmul(mel_filters, magnitudes)

  # Log compression with dynamic range compression
  log_spec <- torch::torch_clamp(mel_spec, min = 1e-10)$log10()
  max_spec <- log_spec$max()
  log_spec <- torch::torch_maximum(log_spec, max_spec - 8.0)
  log_spec <- (log_spec + 4.0) / 4.0

  log_spec
}

# ============================================================================
# S3Tokenizer Model (Simplified)
# ============================================================================

#' S3Tokenizer module
#'
#' This is a simplified implementation. The full S3TokenizerV2 uses a
#' VQ-VAE based on SenseVoice-Large which is complex to port.
#'
#' For now, this provides the mel spectrogram computation and a placeholder
#' for the quantizer that can be loaded from weights.
#'
#' @param n_mels Number of mel bins (default 80)
#' @return nn_module
s3_tokenizer <- torch::nn_module(
  "S3Tokenizer",

  initialize = function(n_mels = 80) {
    self$n_fft <- 400
    self$n_mels <- n_mels

    # Create mel filterbank
    mel_fb <- create_mel_filterbank(
      sr = S3_SR,
      n_fft = self$n_fft,
      n_mels = n_mels,
      fmin = 0,
      fmax = S3_SR / 2
    )
    self$mel_filters <- torch::nn_buffer(
      torch::torch_tensor(mel_fb, dtype = torch::torch_float32())
    )

    # Hann window
    self$window <- torch::nn_buffer(
      torch::torch_hann_window(self$n_fft)
    )

    # Quantizer would go here - this is the complex part
    # The actual model has an encoder + codebook
    # Placeholder for now
    self$encoder <- NULL
    self$codebook <- NULL
  },

  log_mel_spectrogram = function(audio) {
    s3_log_mel_spectrogram(
      audio,
      self$mel_filters,
      self$window,
      self$n_fft,
      device = self$mel_filters$device
    )
  },

  quantize = function(mels, mel_lens) {
    # Placeholder - actual quantization requires the encoder + codebook
    # For now, return dummy tokens

    # The real implementation would:
    # 1. Encode mel spectrograms through the encoder
    # 2. Find nearest codebook entries
    # 3. Return token IDs

    batch_size <- mels$size(1)
    n_frames <- mels$size(3)
    n_tokens <- n_frames %/% 4  # 4 mel frames per token

    # Create placeholder tokens
    tokens <- torch::torch_zeros(c(batch_size, n_tokens),
                                 dtype = torch::torch_long(),
                                 device = mels$device)
    token_lens <- torch::torch_tensor(rep(n_tokens, batch_size),
                                      dtype = torch::torch_long(),
                                      device = mels$device)

    warning("S3Tokenizer quantize() not fully implemented - returning placeholder tokens")

    list(tokens = tokens, lens = token_lens)
  },

  forward = function(wavs, max_len = NULL) {
    device <- self$mel_filters$device

    # Handle input
    if (!inherits(wavs, "torch_tensor")) {
      wavs <- torch::torch_tensor(wavs, dtype = torch::torch_float32())
    }

    if (wavs$dim() == 1) {
      wavs <- wavs$unsqueeze(1)
    }

    wavs <- wavs$to(device = device)

    # Compute mel spectrogram
    mel <- self$log_mel_spectrogram(wavs)

    # Truncate if needed
    if (!is.null(max_len)) {
      mel <- mel[, , 1:min(mel$size(3), max_len * 4)]
    }

    # Quantize
    mel_lens <- torch::torch_tensor(mel$size(3), device = device)
    result <- self$quantize(mel$unsqueeze(1), mel_lens)

    list(
      tokens = result$tokens$squeeze(1),
      lens = result$lens
    )
  }
)

# ============================================================================
# Helper Functions
# ============================================================================

#' Drop invalid speech tokens
#'
#' @param tokens Token tensor
#' @return Filtered tokens
drop_invalid_tokens <- function(tokens) {
  # Remove tokens >= SPEECH_VOCAB_SIZE
  valid_mask <- tokens < SPEECH_VOCAB_SIZE
  tokens[valid_mask]
}

#' Pad audio to multiple of token rate
#'
#' @param wav Audio samples
#' @param sr Sample rate
#' @return Padded audio
pad_audio_for_tokenizer <- function(wav, sr) {
  if (!inherits(wav, "torch_tensor")) {
    wav <- torch::torch_tensor(wav, dtype = torch::torch_float32())
  }

  if (wav$dim() == 1) {
    wav <- wav$unsqueeze(1)
  }

  # Calculate intended length
  n_tokens <- ceiling(wav$size(2) / sr * S3_TOKEN_RATE)
  intended_len <- as.integer(n_tokens * sr / S3_TOKEN_RATE)

  # Pad if needed
  if (wav$size(2) < intended_len) {
    wav <- torch::nnf_pad(wav, c(0, intended_len - wav$size(2)), value = 0)
  }

  wav
}

# ============================================================================
# Weight Loading
# ============================================================================

#' Load S3Tokenizer weights
#'
#' @param model S3Tokenizer model
#' @param state_dict Named list of tensors
#' @return Model with loaded weights
#' @export
load_s3tokenizer_weights <- function(model, state_dict) {
  # The S3Tokenizer weights are part of s3gen.safetensors
  # Need to extract the tokenizer-specific weights

  # Load mel filters if present
  if ("tokenizer._mel_filters" %in% names(state_dict)) {
    model$mel_filters$copy_(state_dict[["tokenizer._mel_filters"]])
  }

  # Load window if present
  if ("tokenizer.window" %in% names(state_dict)) {
    model$window$copy_(state_dict[["tokenizer.window"]])
  }

  # Encoder and codebook weights would go here
  # These are the complex parts of S3TokenizerV2

  model
}
