# Audio utilities for chatteRbox
# Handles audio I/O, resampling, and mel spectrogram computation

#' Read audio file
#'
#' @param path Path to audio file (WAV format)
#' @return List with samples (numeric vector normalized to [-1, 1]) and sr (sample rate)
#' @export
read_audio <- function(path) {
  wav <- tuneR::readWave(path)

  # Extract samples and normalize to [-1, 1]
  if (wav@bit == 16) {
    samples <- wav@left / 32768
  } else if (wav@bit == 24) {
    samples <- wav@left / 8388608
  } else if (wav@bit == 32) {
    samples <- wav@left / 2147483648
  } else {
    samples <- wav@left / (2^(wav@bit - 1))
  }

  list(
    samples = as.numeric(samples),
    sr = wav@samp.rate
  )
}

#' Write audio file
#'
#' @param samples Numeric vector of audio samples (normalized to [-1, 1])
#' @param sr Sample rate
#' @param path Output path (WAV format)
#' @export
write_audio <- function(samples, sr, path) {
  # Handle torch tensor input

if (inherits(samples, "torch_tensor")) {
    samples <- as.numeric(samples$cpu())
  }

  # Clip to valid range
  samples <- pmax(pmin(samples, 0.99), -0.99)

  # Convert to 16-bit integer
  samples_int <- as.integer(samples * 32767)

  wav <- tuneR::Wave(
    left = samples_int,
    samp.rate = as.integer(sr),
    bit = 16
  )

  tuneR::writeWave(wav, path)
}

#' Resample audio
#'
#' @param samples Numeric vector of audio samples
#' @param from_sr Source sample rate
#' @param to_sr Target sample rate
#' @return Resampled audio samples
#' @export
resample_audio <- function(samples, from_sr, to_sr) {
  if (from_sr == to_sr) {
    return(samples)
  }

  # Use linear interpolation for resampling
  # More sophisticated methods could use signal::resample
  n_samples <- length(samples)
  duration <- n_samples / from_sr
  n_new <- as.integer(duration * to_sr)

  old_times <- seq(0, duration, length.out = n_samples)
  new_times <- seq(0, duration, length.out = n_new)

  stats::approx(old_times, samples, new_times, method = "linear")$y
}

#' Create mel filterbank
#'
#' @param sr Sample rate
#' @param n_fft FFT size
#' @param n_mels Number of mel bins
#' @param fmin Minimum frequency
#' @param fmax Maximum frequency
#' @return Mel filterbank matrix (n_mels x (n_fft/2 + 1))
create_mel_filterbank <- function(sr, n_fft, n_mels, fmin = 0, fmax = NULL) {
  if (is.null(fmax)) {
    fmax <- sr / 2
  }

  # Convert Hz to mel scale
  hz_to_mel <- function(hz) {
    2595 * log10(1 + hz / 700)
  }

  mel_to_hz <- function(mel) {
    700 * (10^(mel / 2595) - 1)
  }

  # Create mel points
  mel_min <- hz_to_mel(fmin)
  mel_max <- hz_to_mel(fmax)
  mel_points <- seq(mel_min, mel_max, length.out = n_mels + 2)
  hz_points <- mel_to_hz(mel_points)

  # Convert to FFT bin numbers
  n_fft_bins <- n_fft %/% 2 + 1
  fft_freqs <- seq(0, sr / 2, length.out = n_fft_bins)

  # Create filterbank
  filterbank <- matrix(0, nrow = n_mels, ncol = n_fft_bins)

  for (i in seq_len(n_mels)) {
    left <- hz_points[i]
    center <- hz_points[i + 1]
    right <- hz_points[i + 2]

    # Rising edge
    rising <- (fft_freqs - left) / (center - left)
    rising[fft_freqs < left] <- 0
    rising[fft_freqs > center] <- 0

    # Falling edge
    falling <- (right - fft_freqs) / (right - center)
    falling[fft_freqs < center] <- 0
    falling[fft_freqs > right] <- 0

    filterbank[i, ] <- pmax(rising, 0) + pmax(falling, 0)
    filterbank[i, fft_freqs >= center] <- pmax(falling[fft_freqs >= center], 0)
    filterbank[i, fft_freqs < center] <- pmax(rising[fft_freqs < center], 0)
  }

  filterbank
}

# Cache for mel filterbanks and hann windows
.mel_cache <- new.env(parent = emptyenv())

#' Compute mel spectrogram (S3Gen compatible)
#'
#' @param y Audio samples as torch tensor or numeric vector
#' @param n_fft FFT size (default 1920 for 24kHz)
#' @param n_mels Number of mel bins (default 80)
#' @param sr Sample rate (default 24000)
#' @param hop_size Hop size (default 480)
#' @param win_size Window size (default 1920)
#' @param fmin Minimum frequency (default 0)
#' @param fmax Maximum frequency (default 8000)
#' @param center Whether to center frames (default FALSE)
#' @return Mel spectrogram tensor (batch, n_mels, time)
#' @export
compute_mel_spectrogram <- function(y, n_fft = 1920, n_mels = 80, sr = 24000,
                                     hop_size = 480, win_size = 1920,
                                     fmin = 0, fmax = 8000, center = FALSE) {
  # Convert to torch tensor if needed
  if (!inherits(y, "torch_tensor")) {
    y <- torch::torch_tensor(y, dtype = torch::torch_float32())
  }

  # Add batch dimension if needed
  if (y$dim() == 1) {
    y <- y$unsqueeze(1)
  }

  device <- y$device

  # Get or create mel filterbank
  cache_key <- paste(fmax, device$type, sep = "_")
  if (is.null(.mel_cache[[cache_key]])) {
    mel_fb <- create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    .mel_cache[[cache_key]] <- torch::torch_tensor(mel_fb, dtype = torch::torch_float32())$to(device = device)
  }
  mel_basis <- .mel_cache[[cache_key]]

  # Get or create Hann window
  win_key <- paste("hann", device$type, sep = "_")
  if (is.null(.mel_cache[[win_key]])) {
    .mel_cache[[win_key]] <- torch::torch_hann_window(win_size)$to(device = device)
  }
  hann_window <- .mel_cache[[win_key]]

  # Pad audio (reflect padding)
  pad_amount <- as.integer((n_fft - hop_size) / 2)
  y <- y$unsqueeze(2)  # Add channel dim for padding
  y <- torch::nnf_pad(y, c(pad_amount, pad_amount), mode = "reflect")
  y <- y$squeeze(2)

  # Compute STFT
  spec <- torch::torch_stft(
    y,
    n_fft = n_fft,
    hop_length = hop_size,
    win_length = win_size,
    window = hann_window,
    center = center,
    pad_mode = "reflect",
    normalized = FALSE,
    onesided = TRUE,
    return_complex = TRUE
  )

  # Convert to magnitude
  spec <- torch::torch_view_as_real(spec)
  spec <- torch::torch_sqrt(spec$pow(2)$sum(-1) + 1e-9)

  # Apply mel filterbank
  spec <- torch::torch_matmul(mel_basis, spec)

  # Log compression
  spec <- torch::torch_log(torch::torch_clamp(spec, min = 1e-5))

  spec
}

#' Compute mel spectrogram for voice encoder (40 bins, 16kHz)
#'
#' @param y Audio samples
#' @param sr Sample rate (should be 16000)
#' @return Mel spectrogram (batch, time, 40)
#' @export
compute_mel_spectrogram_ve <- function(y, sr = 16000) {
  # Voice encoder uses different params
  spec <- compute_mel_spectrogram(
    y,
    n_fft = 400,
    n_mels = 40,
    sr = sr,
    hop_size = 160,
    win_size = 400,
    fmin = 0,
    fmax = 8000,
    center = TRUE
  )

  # Transpose to (batch, time, mels) for LSTM
  spec$transpose(2, 3)
}
