# anvl/yunque port of the Chatterbox mel frontends. Torch-free; yq_ marks
# the anvl/XLA implementation.
#
# Two deterministic (weightless) transforms:
#   yq_compute_mel_spectrogram  - 80-bin / 24 kHz S3Gen reference mels
#                                 (librosa/Slaney banks, centered=FALSE with
#                                 manual reflect pad, periodic Hann window)
#   yq_kaldi_fbank              - 80-bin / 16 kHz Kaldi fbank for CAMPPlus
#                                 (snip_edges framing, per-frame DC offset +
#                                 preemphasis, povey window, HTK mel banks,
#                                 log floor eps = 2^-23)
#
# The DFT / power / mel-matmul / log are anvl ops. The per-frame Kaldi
# preprocessing (framing, DC-offset, preemphasis, windowing, zero-pad) is
# done host-side on the raw audio vector - it is deterministic data prep,
# analogous to the host-side embedding index prep in yq_llama, and it does
# not fit yunque::stft's shared-window framing.

# ---- librosa/Slaney mel filterbank (base R, torch-free) --------------------

.yq_create_mel_filterbank <- function(sr, n_fft, n_mels, fmin = 0, fmax = NULL,
                                      norm = "slaney") {
  if (is.null(fmax)) fmax <- sr / 2
  f_sp <- 200.0 / 3
  min_log_hz <- 1000.0
  min_log_mel <- (min_log_hz - 0) / f_sp
  logstep <- log(6.4) / 27.0
  hz_to_mel <- function(hz) {
    ifelse(hz < min_log_hz, hz / f_sp,
      min_log_mel + log(hz / min_log_hz) / logstep)
  }
  mel_to_hz <- function(mel) {
    ifelse(mel < min_log_mel, mel * f_sp,
      min_log_hz * exp(logstep * (mel - min_log_mel)))
  }
  mel_points <- seq(hz_to_mel(fmin), hz_to_mel(fmax), length.out = n_mels + 2)
  hz_points <- mel_to_hz(mel_points)
  n_fft_bins <- n_fft %/% 2 + 1
  fft_freqs <- seq(0, sr / 2, length.out = n_fft_bins)
  fb <- matrix(0, nrow = n_mels, ncol = n_fft_bins)
  for (i in seq_len(n_mels)) {
    left <- hz_points[i]
    center <- hz_points[i + 1]
    right <- hz_points[i + 2]
    rising <- (fft_freqs - left) / (center - left)
    falling <- (right - fft_freqs) / (right - center)
    row <- numeric(n_fft_bins)
    row[fft_freqs < center] <- pmax(rising[fft_freqs < center], 0)
    row[fft_freqs >= center] <- pmax(falling[fft_freqs >= center], 0)
    fb[i, ] <- row
  }
  if (norm == "slaney") {
    enorm <- 2.0 / (hz_points[3:(n_mels + 2)] - hz_points[1:n_mels])
    fb <- fb * enorm
  }
  fb
}

# ---- Kaldi HTK mel filterbank (base R port of get_mel_banks) ---------------

.yq_kaldi_mel_scale <- function(freq) 1127 * log(1 + freq / 700)

.yq_get_mel_banks <- function(num_bins, window_length_padded, sample_freq,
                             low_freq = 20, high_freq = 0) {
  num_fft_bins <- window_length_padded %/% 2
  nyquist <- 0.5 * sample_freq
  if (high_freq <= 0) high_freq <- high_freq + nyquist
  fft_bin_width <- sample_freq / window_length_padded
  mel_low <- .yq_kaldi_mel_scale(low_freq)
  mel_high <- .yq_kaldi_mel_scale(high_freq)
  mel_freq_delta <- (mel_high - mel_low) / (num_bins + 1)
  bin <- 0:(num_bins - 1)
  left_mel <- bin * mel_freq_delta + mel_low            # length num_bins
  center_mel <- (bin + 1) * mel_freq_delta + mel_low
  right_mel <- (bin + 2) * mel_freq_delta + mel_low
  fft_freqs <- (0:(num_fft_bins - 1)) * fft_bin_width
  mel <- .yq_kaldi_mel_scale(fft_freqs)                 # length num_fft_bins
  # [num_bins, num_fft_bins]; left/center/right recycle down columns (len=nrow)
  mel_mat <- matrix(mel, num_bins, num_fft_bins, byrow = TRUE)
  up_slope <- (mel_mat - left_mel) / (center_mel - left_mel)
  down_slope <- (right_mel - mel_mat) / (right_mel - center_mel)
  # pmax(0, m) drops the matrix dim (scalar first arg strips attrs); clamp
  # in place to keep [num_bins, num_fft_bins].
  slope <- pmin(up_slope, down_slope)
  slope[slope < 0] <- 0
  slope
}

.yq_next_power_of_2 <- function(x) {
  if (x == 0) return(1L)
  as.integer(2^ceiling(log2(x)))
}

# torch periodic=FALSE Hann (symmetric, denominator n-1), raised to 0.85.
.yq_povey_window <- function(n) {
  (0.5 - 0.5 * cos(2 * pi * (0:(n - 1L)) / (n - 1L)))^0.85
}

# torch 'reflect' pad of a numeric vector by `pad` each side (mirror without
# repeating the edge sample).
.yq_reflect_pad <- function(x, pad) {
  n <- length(x)
  c(x[(pad + 1L):2L], x, x[(n - 1L):(n - pad)])
}

#' S3Gen reference mel spectrogram (anvl)
#'
#' Torch-free port of \code{compute_mel_spectrogram}: manual reflect pad,
#' periodic-Hann STFT (\code{center = FALSE}) via \code{yunque::stft}, power
#' magnitude, Slaney mel filterbank matmul, log-clamp.
#'
#' @param y Numeric audio vector.
#' @param n_fft,n_mels,sr,hop_size,win_size,fmin,fmax Frontend params
#'   (defaults match the 24 kHz S3Gen config).
#' @param center Centered STFT (torch \code{center = TRUE} semantics: an
#'   extra \code{n_fft/2} reflect pad on top of the fixed frontend pad).
#'
#' @return AnvlArray \code{[1, n_mels, n_frames]}.
#'
#' @export
yq_compute_mel_spectrogram <- function(y, n_fft = 1920L, n_mels = 80L,
                                       sr = 24000L, hop_size = 480L,
                                       win_size = 1920L, fmin = 0,
                                       fmax = 8000, center = FALSE) {
  y <- as.numeric(y)
  pad_amount <- as.integer((n_fft - hop_size) / 2)
  y <- .yq_reflect_pad(y, pad_amount)
  if (center) {
    # torch_stft center=TRUE reflect-pads the already-padded signal
    y <- .yq_reflect_pad(y, n_fft %/% 2L)
  }
  sig <- anvl::nv_array(matrix(y, nrow = 1L), dtype = "f32")
  win <- yunque::hann_window(win_size)
  sp <- yunque::stft(sig, n_fft = as.integer(n_fft),
    hop_length = as.integer(hop_size), window = win, center = FALSE)
  nf <- n_fft %/% 2L + 1L
  nframes <- anvl::shape(sp$real)[3L]
  # magnitude = sqrt(re^2 + im^2 + 1e-9)
  mag <- anvl::nv_sqrt(anvl::nv_add(sp$real * sp$real + sp$imag * sp$imag, 1e-9))
  spec2d <- anvl::nv_reshape(mag, c(nf, nframes))       # batch = 1
  mel_fb <- .yq_create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
  mel_basis <- anvl::nv_array(mel_fb, dtype = "f32")    # [n_mels, nf]
  spec <- anvl::nv_matmul(mel_basis, spec2d)            # [n_mels, nframes]
  spec <- anvl::nv_log(anvl::nv_max(spec, 1e-5))
  anvl::nv_reshape(spec, c(1L, n_mels, nframes))
}

#' Voice-encoder mel spectrogram (anvl)
#'
#' Torch-free port of \code{compute_mel_spectrogram_ve}: 16 kHz, 40 bins,
#' centered STFT, transposed to \code{[1, T, 40]} for the LSTM.
#'
#' @param y Numeric audio vector (16 kHz).
#' @param sr Sample rate (default 16000).
#'
#' @return AnvlArray \code{[1, n_frames, 40]}.
#'
#' @export
yq_compute_mel_spectrogram_ve <- function(y, sr = 16000) {
  spec <- yq_compute_mel_spectrogram(y, n_fft = 400L, n_mels = 40L, sr = sr,
    hop_size = 160L, win_size = 400L, fmin = 0, fmax = 8000, center = TRUE)
  anvl::nv_transpose(spec, c(1L, 3L, 2L))
}

#' S3 tokenizer log-mel spectrogram (anvl)
#'
#' Torch-free port of \code{s3_log_mel_spectrogram} (whisper-style): centered
#' Hann STFT with the last frame dropped, power spectrum, Slaney mel banks,
#' \code{log10} with an 8-dB dynamic-range floor, and \code{(x + 4) / 4}
#' scaling. The floor/scale run host-side (global max reduce).
#'
#' @param audio Numeric audio vector (16 kHz).
#' @param n_mels Mel bins (default 128).
#' @param sr,n_fft,hop STFT params (defaults match the S3 tokenizer).
#'
#' @return AnvlArray \code{[1, n_mels, n_frames]}.
#'
#' @export
yq_s3_log_mel_spectrogram <- function(audio, n_mels = 128L, sr = 16000L,
                                      n_fft = 400L, hop = 160L) {
  y <- .yq_reflect_pad(as.numeric(audio), n_fft %/% 2L) # torch center=TRUE
  sig <- anvl::nv_array(matrix(y, nrow = 1L), dtype = "f32")
  win <- yunque::hann_window(n_fft)
  sp <- yunque::stft(sig, n_fft = as.integer(n_fft),
    hop_length = as.integer(hop), window = win, center = FALSE)
  nf <- n_fft %/% 2L + 1L
  nframes <- anvl::shape(sp$real)[3L]
  mag2 <- sp$real * sp$real + sp$imag * sp$imag
  mag2 <- yunque::slice_lastdim(mag2, 1L, nframes - 1L) # whisper drops last
  spec2d <- anvl::nv_reshape(mag2, c(nf, nframes - 1L)) # batch = 1
  mel_fb <- .yq_create_mel_filterbank(sr, n_fft, n_mels, 0, sr / 2)
  spec <- anvl::nv_matmul(anvl::nv_array(mel_fb, dtype = "f32"), spec2d)
  # log10 + dynamic-range floor + scale, host-side (global max reduce)
  s <- log10(pmax(as.array(spec), 1e-10))
  s <- pmax(s, max(s) - 8)
  anvl::nv_array(array((s + 4) / 4, c(1L, n_mels, nframes - 1L)), dtype = "f32")
}

#' Kaldi log-mel fbank for CAMPPlus (anvl)
#'
#' Torch-free port of \code{kaldi_fbank} (torchaudio compliance defaults used
#' by CAMPPlus): snip_edges framing, per-frame DC-offset and 0.97 preemphasis,
#' povey window, power spectrum via an anvl DFT matmul, HTK mel banks, and a
#' \code{log(max(x, 2^-23))} floor.
#'
#' @param audio Numeric audio vector.
#' @param num_mel_bins Number of triangular mel bins (default 80).
#' @param sample_rate Sample rate in Hz (default 16000).
#'
#' @return AnvlArray \code{[n_frames, num_mel_bins]}.
#'
#' @export
yq_kaldi_fbank <- function(audio, num_mel_bins = 80L, sample_rate = 16000) {
  wav <- as.numeric(audio)
  n <- length(wav)
  frame_shift_ms <- 10
  frame_length_ms <- 25
  window_shift <- as.integer(sample_rate * frame_shift_ms * 0.001)  # 160
  window_size <- as.integer(sample_rate * frame_length_ms * 0.001)  # 400
  padded <- .yq_next_power_of_2(window_size)                        # 512
  if (n < window_size) stop("waveform is shorter than one window")

  # snip_edges framing: m = 1 + (n - window_size) %/% window_shift
  m <- 1L + (n - window_size) %/% window_shift
  starts <- (0:(m - 1L)) * window_shift
  frames <- matrix(0, m, window_size)
  for (i in seq_len(m)) frames[i, ] <- wav[(starts[i] + 1L):(starts[i] + window_size)]

  # remove DC offset (per-frame mean)
  frames <- frames - rowMeans(frames)

  # preemphasis: x[,j] -= 0.97 * x[,max(1,j-1)] (replicate pad at j=1)
  coeff <- 0.97
  prev <- cbind(frames[, 1L], frames[, 1:(window_size - 1L)])
  frames <- frames - coeff * prev

  # povey window
  frames <- sweep(frames, 2L, .yq_povey_window(window_size), `*`)

  # zero-pad columns 400 -> 512
  framed <- matrix(0, m, padded)
  framed[, 1:window_size] <- frames

  # power spectrum via anvl DFT matmul: X[k]=sum_n x[n] exp(-2pi i k n / N)
  nf <- padded %/% 2L + 1L                                          # 257
  nn <- 0:(padded - 1L)
  kk <- 0:(nf - 1L)
  ang <- outer(nn, kk, function(a, b) 2 * pi * a * b / padded)      # [512, 257]
  cos_basis <- anvl::nv_array(cos(ang), dtype = "f32")
  sin_basis <- anvl::nv_array(-sin(ang), dtype = "f32")
  x <- anvl::nv_array(framed, dtype = "f32")                        # [m, 512]
  re <- anvl::nv_matmul(x, cos_basis)                               # [m, 257]
  im <- anvl::nv_matmul(x, sin_basis)
  power <- re * re + im * im                                        # [m, 257]

  # HTK mel banks [num_mel_bins, 256] -> pad right zero col -> [num, 257]
  banks <- .yq_get_mel_banks(num_mel_bins, padded, sample_rate)
  banks_padded <- cbind(banks, 0)                                   # [num, 257]
  mel_basis <- anvl::nv_array(t(banks_padded), dtype = "f32")       # [257, num]
  mel_energies <- anvl::nv_matmul(power, mel_basis)                 # [m, num]

  # use_log_fbank with eps floor
  anvl::nv_log(anvl::nv_max(mel_energies, 2^(-23)))
}
