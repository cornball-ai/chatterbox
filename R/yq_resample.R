# anvl/yunque port of the windowed-sinc resampler (torchaudio
# sinc_interp_hann): host-side float64 kernel construction, one strided
# yunque::conv1d for the polyphase application.

.yq_gcd <- function(a, b) {
  a <- as.integer(a)
  b <- as.integer(b)
  while (b != 0L) {
    tmp <- b
    b <- a %% b
    a <- tmp
  }
  a
}

# torchaudio _get_sinc_resample_kernel on gcd-reduced rates: float64 math,
# cast to f32 by the caller (matching the torchaudio kernel cache).
# Returns list(kernel = [new_freq, kernel_width] matrix, width).
.yq_sinc_kernel <- function(orig_freq, new_freq, lowpass_filter_width = 6L,
                            rolloff = 0.99) {
  base_freq <- min(orig_freq, new_freq) * rolloff
  width <- as.integer(ceiling(lowpass_filter_width * orig_freq / base_freq))
  idx <- seq.int(-width, width + orig_freq - 1L) / orig_freq # [kw]
  phase <- -(seq_len(new_freq) - 1) / new_freq # [new_freq]
  t <- outer(phase, idx, "+") * base_freq # [new_freq, kw]
  t <- pmin(pmax(t, -lowpass_filter_width), lowpass_filter_width)
  window <- cos(t * pi / lowpass_filter_width / 2)^2
  tp <- t * pi
  kern <- ifelse(tp == 0, 1, sin(tp) / tp) * window * (base_freq / orig_freq)
  list(kernel = kern, width = width)
}

#' Windowed-sinc resample (anvl)
#'
#' Torch-free port of \code{sinc_resample} (torchaudio
#' \code{sinc_interp_hann}): the kernel is built host-side in float64 and
#' cast to f32 like the torchaudio kernel cache, then applied as one
#' strided \code{yunque::conv1d} with the polyphase outputs interleaved.
#'
#' @param audio Numeric audio vector.
#' @param orig_sr,new_sr Sample rates.
#' @param lowpass_filter_width Filter sharpness, more is sharper (default 6).
#' @param rolloff Roll-off frequency as a fraction of Nyquist (default 0.99).
#'
#' @return Numeric vector at \code{new_sr}.
#'
#' @export
yq_resample <- function(audio, orig_sr, new_sr, lowpass_filter_width = 6L,
                        rolloff = 0.99) {
  y <- as.numeric(audio)
  if (orig_sr == new_sr) {
    return(y)
  }
  g <- .yq_gcd(orig_sr, new_sr)
  of <- as.integer(orig_sr) %/% g
  nf <- as.integer(new_sr) %/% g
  kw <- .yq_sinc_kernel(of, nf, lowpass_filter_width, rolloff)
  len <- length(y)
  y <- c(rep(0, kw$width), y, rep(0, kw$width + of))
  x <- anvl::nv_array(array(y, c(1L, 1L, length(y))), dtype = "f32")
  k <- anvl::nv_array(array(kw$kernel, c(nf, 1L, ncol(kw$kernel))),
    dtype = "f32")
  res <- yunque::conv1d(x, k, stride = of) # [1, new_freq, frames]
  frames <- anvl::shape(res)[3L]
  out <- anvl::nv_reshape(anvl::nv_transpose(res, c(1L, 3L, 2L)),
    c(1L, frames * nf)) # interleave the polyphase outputs
  target <- as.integer(ceiling(nf * len / of))
  as.numeric(as.array(out))[seq_len(target)]
}
