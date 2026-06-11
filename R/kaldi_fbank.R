# Kaldi-compatible log mel filterbank features
# Port of torchaudio.compliance.kaldi.fbank
# Reference: torchaudio 2.7.0 compliance/kaldi.py
# Defaults match the CAMPPlus call: Kaldi.fbank(wav, num_mel_bins = 80)

# std::numeric_limits<float>::epsilon() = 2^-23
.kaldi_epsilon <- 2 ^ (- 23)

#' Smallest power of 2 greater than or equal to x
#'
#' @param x Positive integer
#' @return Smallest power of 2 >= x
#' @noRd
.next_power_of_2 <- function (x)
{
    if (x == 0) {
        return(1L)
    }
    as.integer(2 ^ ceiling(log2(x)))
}

#' Kaldi HTK mel scale (Hz to mel)
#'
#' @param freq Frequency in Hz (scalar or tensor)
#' @return Frequency in mels
#' @noRd
.kaldi_mel_scale <- function (freq)
{
    if (inherits(freq, "torch_tensor")) {
        freq$div(700)$add(1)$log()$mul(1127)
    } else {
        1127 * log(1 + freq / 700)
    }
}

#' Kaldi mel filterbank matrix
#'
#' Port of torchaudio get_mel_banks (vtln_warp_factor = 1 path only).
#' Computed in float32 to match torchaudio's default dtype promotion.
#'
#' @param num_bins Number of triangular mel bins
#' @param window_length_padded Padded window size (power of 2)
#' @param sample_freq Sample rate in Hz
#' @param low_freq Low cutoff frequency (default 20)
#' @param high_freq High cutoff frequency; if <= 0, offset from Nyquist
#' @return Tensor (num_bins, window_length_padded / 2)
#' @noRd
.get_mel_banks <- function (num_bins, window_length_padded, sample_freq,
                            low_freq = 20, high_freq = 0)
{
    stopifnot(num_bins > 3, window_length_padded %% 2 == 0)
    num_fft_bins <- window_length_padded %/% 2
    nyquist <- 0.5 * sample_freq

    if (high_freq <= 0) {
        high_freq <- high_freq + nyquist
    }
    stopifnot(low_freq >= 0, low_freq < nyquist,
        high_freq > 0, high_freq <= nyquist, low_freq < high_freq)

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width <- sample_freq / window_length_padded
    mel_low_freq <- .kaldi_mel_scale(low_freq)
    mel_high_freq <- .kaldi_mel_scale(high_freq)

    # divide by num_bins + 1 because of end-effects where the bins
    # spread out to the sides
    mel_freq_delta <- (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin <- torch::torch_arange(
        0, num_bins - 1,
        dtype = torch::torch_float32()
    )$unsqueeze(2) # (num_bins, 1)
    left_mel <- bin$mul(mel_freq_delta)$add(mel_low_freq)
    center_mel <- (bin + 1)$mul(mel_freq_delta)$add(mel_low_freq)
    right_mel <- (bin + 2)$mul(mel_freq_delta)$add(mel_low_freq)

    # (1, num_fft_bins)
    fft_freqs <- torch::torch_arange(
        0, num_fft_bins - 1,
        dtype = torch::torch_float32()
    )$mul(fft_bin_width)
    mel <- .kaldi_mel_scale(fft_freqs)$unsqueeze(1)

    # (num_bins, num_fft_bins)
    up_slope <- (mel - left_mel)$div(center_mel - left_mel)
    down_slope <- (right_mel - mel)$div(right_mel - center_mel)

    # left_mel < center_mel < right_mel: min the slopes, clamp negatives
    zero <- torch::torch_zeros(1)
    torch::torch_max(zero, other = torch::torch_min(up_slope, other = down_slope))
}

#' Povey window
#'
#' Like Hann but goes to zero at the edges (hann ^ 0.85).
#'
#' @param window_size Window length in samples
#' @return Float32 tensor of length window_size
#' @noRd
.povey_window <- function (window_size)
{
    torch::torch_hann_window(
        window_size,
        periodic = FALSE,
        dtype = torch::torch_float32()
    )$pow(0.85)
}

#' Frame, preprocess, and window a waveform (Kaldi-style)
#'
#' Port of torchaudio _get_window for the snip_edges = TRUE,
#' dither = 0 path.
#'
#' @param waveform 1D float tensor
#' @param padded_window_size Window size padded to a power of 2
#' @param window_size Window length in samples
#' @param window_shift Frame shift in samples
#' @param remove_dc_offset Subtract per-frame mean (default TRUE)
#' @param preemphasis_coefficient Preemphasis coefficient (default 0.97)
#' @return Tensor (m, padded_window_size)
#' @noRd
.kaldi_get_window <- function (waveform, padded_window_size, window_size,
                               window_shift, remove_dc_offset = TRUE,
                               preemphasis_coefficient = 0.97)
{
    num_samples <- waveform$size(1)
    if (num_samples < window_size) {
        stop("waveform is shorter than one window")
    }

    # snip_edges framing: m = 1 + (n - window_size) %/% window_shift
    # Note: $unfold() with a positive dim is off-by-one in CRAN torch;
    # -1 (last dim) is handled correctly.
    strided_input <- waveform$unfold(- 1, window_size, window_shift)

    if (remove_dc_offset) {
        # Subtract each frame by its mean
        row_means <- strided_input$mean(dim = 2, keepdim = TRUE)
        strided_input <- strided_input - row_means
    }

    if (preemphasis_coefficient != 0) {
        # x[i, j] -= coeff * x[i, max(0, j - 1)]
        offset_strided_input <- torch::nnf_pad(
            strided_input$unsqueeze(1), c(1, 0),
            mode = "replicate"
        )$squeeze(1) # (m, window_size + 1)
        strided_input <- strided_input -
            offset_strided_input[, 1:window_size]$mul(preemphasis_coefficient)
    }

    # Apply window function to each frame
    window_function <- .povey_window(window_size)$unsqueeze(1)
    strided_input <- strided_input * window_function

    # Pad columns with zeros up to padded_window_size
    if (padded_window_size != window_size) {
        padding_right <- padded_window_size - window_size
        strided_input <- torch::nnf_pad(
            strided_input, c(0, padding_right),
            mode = "constant", value = 0
        )
    }

    strided_input
}

#' Kaldi-compatible log mel filterbank features
#'
#' Port of torchaudio.compliance.kaldi.fbank with the defaults used by
#' CAMPPlus: frame_length = 25 ms, frame_shift = 10 ms, dither = 0,
#' preemphasis 0.97, remove_dc_offset, round_to_power_of_two,
#' snip_edges, povey window, low_freq = 20, high_freq = Nyquist,
#' use_power, use_log_fbank, no energy, no VTLN.
#'
#' @param audio Numeric vector or 1D torch tensor at sample_rate
#' @param num_mel_bins Number of triangular mel bins (default 80)
#' @param sample_rate Sample rate in Hz (default 16000)
#' @return Float32 tensor (n_frames, num_mel_bins)
#' @noRd
kaldi_fbank <- function (audio, num_mel_bins = 80L, sample_rate = 16000)
{
    if (!inherits(audio, "torch_tensor")) {
        audio <- torch::torch_tensor(audio, dtype = torch::torch_float32())
    }
    waveform <- audio$reshape(- 1)

    frame_shift <- 10 # ms
    frame_length <- 25 # ms
    window_shift <- as.integer(sample_rate * frame_shift * 0.001)
    window_size <- as.integer(sample_rate * frame_length * 0.001)
    padded_window_size <- .next_power_of_2(window_size)

    # (m, padded_window_size)
    strided_input <- .kaldi_get_window(
        waveform, padded_window_size, window_size, window_shift
    )

    # Power spectrum, size (m, padded_window_size / 2 + 1)
    spectrum <- torch::torch_fft_rfft(strided_input)$abs()$pow(2)

    # (num_mel_bins, padded_window_size / 2)
    mel_energies <- .get_mel_banks(
        num_mel_bins, padded_window_size, sample_rate
    )$to(device = waveform$device)

    # Pad right column with zeros: (num_mel_bins, padded_window_size / 2 + 1)
    mel_energies <- torch::nnf_pad(
        mel_energies, c(0, 1),
        mode = "constant", value = 0
    )

    # Sum mel filterbanks over the power spectrum: (m, num_mel_bins)
    mel_energies <- torch::torch_mm(spectrum, mel_energies$t())

    # use_log_fbank: avoid log of zero
    eps <- torch::torch_tensor(.kaldi_epsilon, dtype = mel_energies$dtype)
    mel_energies <- torch::torch_max(mel_energies, other = eps)$log()

    mel_energies
}
