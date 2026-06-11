# Windowed-sinc resampler
# Port of torchaudio.functional.resample (sinc_interp_hann method)
# Reference: torchaudio 2.7.0 functional/functional.py
#   _get_sinc_resample_kernel / _apply_sinc_resample_kernel

#' Greatest common divisor
#'
#' @param a Integer scalar
#' @param b Integer scalar
#' @return Greatest common divisor of a and b
#' @noRd
.gcd <- function (a, b)
{
    a <- as.integer(a)
    b <- as.integer(b)
    while (b != 0L) {
        tmp <- b
        b <- a %% b
        a <- tmp
    }
    a
}

#' Build windowed-sinc resampling kernel
#'
#' Port of torchaudio _get_sinc_resample_kernel. Kernel is computed in
#' float64 then cast to float32, matching torchaudio.transforms.Resample
#' defaults (dtype = NULL).
#'
#' @param orig_freq Original sample rate (integer)
#' @param new_freq Target sample rate (integer)
#' @param gcd Greatest common divisor of the two rates
#' @param lowpass_filter_width Filter sharpness (default 6)
#' @param rolloff Roll-off frequency as fraction of Nyquist (default 0.99)
#' @return List with kernel tensor (new_freq, 1, kernel_width) and width
#' @noRd
.get_sinc_resample_kernel <- function (orig_freq, new_freq, gcd,
                                       lowpass_filter_width = 6L,
                                       rolloff = 0.99)
{
    orig_freq <- as.integer(orig_freq) %/% gcd
    new_freq <- as.integer(new_freq) %/% gcd

    if (lowpass_filter_width <= 0) {
        stop("Low pass filter width should be positive.")
    }

    base_freq <- min(orig_freq, new_freq)
    # Antialiasing: remove the highest frequencies
    base_freq <- base_freq * rolloff

    width <- ceiling(lowpass_filter_width * orig_freq / base_freq)

    # Python: torch.arange(-width, width + orig_freq) is exclusive of the
    # endpoint; R torch_arange is inclusive, so stop one short.
    idx <- torch::torch_arange(
        - width, width + orig_freq - 1,
        dtype = torch::torch_float64()
    )
    idx <- idx$unsqueeze(1)$unsqueeze(1)$div(orig_freq) # (1, 1, kw)

    # Python: torch.arange(0, -new_freq, -1) -> 0, -1, ..., -(new_freq - 1)
    t <- torch::torch_arange(
        0, new_freq - 1,
        dtype = torch::torch_float64()
    )$neg()
    t <- t$unsqueeze(2)$unsqueeze(3)$div(new_freq) # (new_freq, 1, 1)
    t <- t$add(idx) # (new_freq, 1, kw)
    t <- t$mul(base_freq)
    t <- t$clamp(- lowpass_filter_width, lowpass_filter_width)

    # Hann window evaluated at the sample positions
    window <- t$mul(pi / lowpass_filter_width / 2)$cos()$pow(2)

    t <- t$mul(pi)

    scale <- base_freq / orig_freq
    one <- torch::torch_ones_like(t)
    kernels <- torch::torch_where(t == 0, one, t$sin()$div(t))
    kernels <- kernels$mul(window)$mul(scale)

    # transforms.Resample caches the kernel as float32
    kernels <- kernels$to(dtype = torch::torch_float32())

    list(kernel = kernels, width = width)
}

#' Apply windowed-sinc resampling kernel
#'
#' Port of torchaudio _apply_sinc_resample_kernel for a single waveform.
#'
#' @param waveform Float tensor (1, time)
#' @param orig_freq Original sample rate (integer)
#' @param new_freq Target sample rate (integer)
#' @param gcd Greatest common divisor of the two rates
#' @param kernel Resampling kernel (new_freq, 1, kernel_width)
#' @param width Filter half-width
#' @return Resampled tensor (1, new_time)
#' @noRd
.apply_sinc_resample_kernel <- function (waveform, orig_freq, new_freq, gcd,
                                         kernel, width)
{
    orig_freq <- as.integer(orig_freq) %/% gcd
    new_freq <- as.integer(new_freq) %/% gcd

    len <- waveform$size(2)
    waveform <- torch::nnf_pad(waveform, c(width, width + orig_freq))
    resampled <- torch::nnf_conv1d(
        waveform$unsqueeze(2), kernel,
        stride = orig_freq
    ) # (1, new_freq, frames)
    resampled <- resampled$transpose(2, 3)$reshape(c(1, - 1))
    target_length <- as.integer(ceiling(new_freq * as.numeric(len) / orig_freq))
    resampled[, 1:target_length]
}

#' Resample audio with bandlimited sinc interpolation
#'
#' Port of torchaudio.functional.resample with the sinc_interp_hann
#' window. Produces output identical to
#' torchaudio.transforms.Resample(orig_sr, new_sr) defaults.
#'
#' @param audio Numeric vector or 1D torch tensor
#' @param orig_sr Original sample rate (integer)
#' @param new_sr Target sample rate (integer)
#' @param lowpass_filter_width Filter sharpness, more is sharper (default 6)
#' @param rolloff Roll-off frequency as fraction of Nyquist (default 0.99)
#' @param resampling_method Only "sinc_interp_hann" is supported
#' @return Resampled audio, same type as input (numeric vector in,
#'   numeric vector out; tensor in, 1D tensor out)
#' @noRd
sinc_resample <- function (audio, orig_sr, new_sr, lowpass_filter_width = 6L,
                           rolloff = 0.99,
                           resampling_method = "sinc_interp_hann")
{
    if (resampling_method != "sinc_interp_hann") {
        stop("Only resampling_method = 'sinc_interp_hann' is supported")
    }
    if (orig_sr <= 0 || new_sr <= 0) {
        stop("Original frequency and desired frequency should be positive")
    }

    was_numeric <- !inherits(audio, "torch_tensor")

    if (orig_sr == new_sr) {
        return(audio)
    }

    if (was_numeric) {
        waveform <- torch::torch_tensor(audio, dtype = torch::torch_float32())
    } else {
        waveform <- audio
    }
    waveform <- waveform$reshape(c(1, - 1))

    g <- .gcd(orig_sr, new_sr)
    kw <- .get_sinc_resample_kernel(
        orig_sr, new_sr, g,
        lowpass_filter_width = lowpass_filter_width,
        rolloff = rolloff
    )
    kernel <- kw$kernel$to(device = waveform$device)

    resampled <- .apply_sinc_resample_kernel(
        waveform, orig_sr, new_sr, g, kernel, kw$width
    )
    resampled <- resampled$squeeze(1)

    if (was_numeric) {
        as.numeric(resampled$cpu())
    } else {
        resampled
    }
}
