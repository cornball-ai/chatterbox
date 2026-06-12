# ITU-R BS.1770-4 loudness measurement and normalization (mono).
# Port of pyloudnorm's K-weighting Meter as used by Python chatterbox
# turbo's norm_loudness() (tts_turbo.py). Pure base R: two RBJ biquads
# via stats::filter, gated 400 ms blocks at 75 % overlap.

# RBJ cookbook biquad coefficients (Audio-EQ-Cookbook), matching
# pyloudnorm's IIRfilter.generate_coefficients for the two K-weighting
# stages: high shelf (G = 4 dB, Q = 1/sqrt(2), fc = 1500) and high pass
# (Q = 0.5, fc = 38).
.k_weighting_coeffs <- function (sample_rate) {
    rbj <- function (G, Q, fc, type) {
        A <- 10^(G / 40)
        w0 <- 2 * pi * fc / sample_rate
        alpha <- sin(w0) / (2 * Q)
        if (type == "high_shelf") {
            b <- c(A * ((A + 1) + (A - 1) * cos(w0) + 2 * sqrt(A) * alpha),
                   -2 * A * ((A - 1) + (A + 1) * cos(w0)),
                   A * ((A + 1) + (A - 1) * cos(w0) - 2 * sqrt(A) * alpha))
            a <- c((A + 1) - (A - 1) * cos(w0) + 2 * sqrt(A) * alpha,
                   2 * ((A - 1) - (A + 1) * cos(w0)),
                   (A + 1) - (A - 1) * cos(w0) - 2 * sqrt(A) * alpha)
        } else { # high_pass
            b <- c((1 + cos(w0)) / 2, -(1 + cos(w0)), (1 + cos(w0)) / 2)
            a <- c(1 + alpha, -2 * cos(w0), 1 - alpha)
        }
        list(b = b, a = a)
    }
    list(rbj(4.0, 1 / sqrt(2), 1500.0, "high_shelf"),
         rbj(0.0, 0.5, 38.0, "high_pass"))
}

# scipy.signal.lfilter for a biquad: zero initial state, normalized by
# a[1]. FIR part vectorized, IIR part via stats::filter(recursive).
.biquad <- function (x, b, a) {
    b <- b / a[1]
    a <- a / a[1]
    v <- b[1] * x +
        b[2] * c(0, x[-length(x)]) +
        b[3] * c(0, 0, x[-((length(x) - 1):length(x))])
    y <- stats::filter(v, filter = -a[2:3], method = "recursive",
        init = c(0, 0))
    as.numeric(y)
}

#' Integrated loudness (ITU-R BS.1770-4)
#'
#' Measures the integrated gated loudness of a mono signal in LUFS,
#' matching pyloudnorm's K-weighting meter (the measurement Python
#' chatterbox turbo applies to reference audio).
#'
#' @param samples Numeric vector of mono audio samples.
#' @param sample_rate Sample rate in Hz.
#' @return Loudness in LUFS (\code{-Inf} for silence or when no block
#'   passes the gates).
#' @export
integrated_loudness <- function (samples, sample_rate) {
    if (!is.numeric(samples) || length(samples) < 0.4 * sample_rate) {
        stop("samples must be numeric mono audio of at least 400 ms")
    }
    x <- as.numeric(samples)
    for (f in .k_weighting_coeffs(sample_rate)) {
        x <- .biquad(x, f$b, f$a)
    }

    # Gated 400 ms blocks, 75 % overlap (step 100 ms); bounds and block
    # count follow pyloudnorm exactly (int() truncation, round at end)
    t_g <- 0.4
    step <- 0.25
    n <- length(x)
    n_blocks <- as.integer(round((n / sample_rate - t_g) / (t_g * step))) + 1L
    j <- 0:(n_blocks - 1)
    lower <- trunc(t_g * j * step * sample_rate)
    upper <- trunc(t_g * (j * step + 1) * sample_rate)
    cs <- c(0, cumsum(x^2))
    z <- (cs[upper + 1] - cs[lower + 1]) / (t_g * sample_rate)
    l <- -0.691 + 10 * log10(z)

    # Absolute gate (-70 LUFS), then relative gate (10 LU below the
    # mean of absolutely-gated blocks)
    abs_gated <- which(l >= -70)
    if (length(abs_gated) == 0) {
        return(-Inf)
    }
    gamma_r <- -0.691 + 10 * log10(mean(z[abs_gated])) - 10
    gated <- which(l > gamma_r & l > -70)
    if (length(gated) == 0) {
        return(-Inf)
    }
    -0.691 + 10 * log10(mean(z[gated]))
}

#' Normalize audio to a target loudness
#'
#' Applies a constant gain so the signal measures \code{target_lufs}
#' integrated loudness. Mirrors Python chatterbox turbo's
#' \code{norm_loudness()}: when the gain is non-finite or non-positive
#' (e.g. silence), the input is returned unchanged.
#'
#' @param samples Numeric vector of mono audio samples.
#' @param sample_rate Sample rate in Hz.
#' @param target_lufs Target integrated loudness (default -27, the
#'   Python turbo conditioning default).
#' @return Gain-adjusted samples.
#' @export
normalize_loudness <- function (samples, sample_rate, target_lufs = -27) {
    loudness <- tryCatch(integrated_loudness(samples, sample_rate),
        error = function (e) NA_real_)
    if (is.na(loudness)) {
        return(samples)
    }
    gain <- 10^((target_lufs - loudness) / 20)
    if (!is.finite(gain) || gain <= 0) {
        return(samples)
    }
    samples * gain
}
