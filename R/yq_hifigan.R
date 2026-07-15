# anvl/yunque port of the HiFT/HiFiGAN vocoder (NSF + ISTFTNet):
# mel [B, 80, T] -> waveform [B, T * 480]. Torch-free; yq_ marks the
# anvl/XLA implementation. The harmonic source (F0 -> phase cumsum ->
# sine bank + noise) is prepared host-side: torch's CPU float32 cumsum
# accumulates in double and rounds each partial sum to float32 on
# output, which host R emulates exactly (verified bitwise), while the
# heavy conv/ISTFT math stays in anvl. Stochastic draws (harmonic
# initial phase, source noise) are explicit arguments so parity
# fixtures can inject the reference RNG values.

.yq_leaky_relu <- function(x, slope = 0.01) {
  anvl::nv_max(x, 0) + anvl::nv_min(x, 0) * slope
}

# Round doubles to their nearest float32 values (kept as doubles).
# Used to mirror the reference's float32 elementwise rounding in the
# host-side source prep; single-rounding emulation via double is exact
# for +, -, *, /.
.yq_f32 <- function(x) {
  d <- dim(x)
  y <- readBin(writeBin(as.numeric(x), raw(), size = 4L), "numeric",
    size = 4L, n = length(x))
  if (!is.null(d)) {
    dim(y) <- d
  }
  y
}

# Static slice along one dim of a rank-3 array (1-based inclusive).
.yq_slice3 <- function(x, dim, from, to) {
  s <- anvl::shape(x)
  start <- rep(1L, 3L)
  start[dim] <- as.integer(from)
  lim <- s
  lim[dim] <- as.integer(to)
  anvl::nv_static_slice(x, start_indices = start, limit_indices = lim,
    strides = rep(1L, 3L))
}

# Reflection padding on the last dim of [B, C, L] (torch 'reflect':
# mirror without repeating the edge sample).
.yq_reflect_pad1d <- function(x, left, right) {
  s <- anvl::shape(x)
  L <- s[3L]
  parts <- list()
  if (left > 0L) {
    parts[[length(parts) + 1L]] <- anvl::nv_reverse(
      .yq_slice3(x, 3L, 2L, left + 1L), dims = 3L)
  }
  parts[[length(parts) + 1L]] <- x
  if (right > 0L) {
    parts[[length(parts) + 1L]] <- anvl::nv_reverse(
      .yq_slice3(x, 3L, L - right, L - 1L), dims = 3L)
  }
  if (length(parts) == 1L) {
    return(x)
  }
  do.call(anvl::nv_concatenate, c(parts, list(dimension = 3L)))
}

# HiFiGAN resblock: snake/conv(dilated)/snake/conv branches with
# residual adds. b carries $kernel and $branches (dilation, alphas as
# [1, C, 1], weight-normed conv weights). Snake here is the
# single-parameter form x + sin^2(alpha x) / (alpha + 1e-9)
# (alpha_logscale = FALSE), which is yunque::snake with beta = alpha.
.yq_hifigan_resblock <- function(x, b) {
  k <- b$kernel
  for (br in b$branches) {
    d <- br$dilation
    xt <- yunque::snake(x, br$alpha1)
    xt <- yunque::conv1d(xt, br$conv1_w, br$conv1_b,
      padding = (k * d - d) %/% 2L, dilation = d)
    xt <- yunque::snake(xt, br$alpha2)
    xt <- yunque::conv1d(xt, br$conv2_w, br$conv2_b,
      padding = (k - 1L) %/% 2L)
    x <- x + xt
  }
  x
}

# NSF sine bank, host side. f0_up: R matrix [B, T_wav] of float32-exact
# values (nearest-upsampled F0 in Hz). phase: [B, H+1, 1] initial phase
# draws in (-pi, pi) (the fundamental is forced to 0 here, as in the
# reference). z: [B, H+1, T_wav] standard-normal draws. Every float32
# op of the reference is emulated by rounding through .yq_f32; sin()
# itself is host double (<= 1 ulp from torch's float32 sin). Returns
# sine waves [B, T_wav, H+1] laid out for the harmonics linear.
.yq_hifigan_sine <- function(f0_up, phase, z, sample_rate = 24000,
                             harmonic_num = 8L, sine_amp = 0.1,
                             noise_std = 0.003, voiced_threshold = 10) {
  B <- nrow(f0_up)
  TT <- ncol(f0_up)
  H <- harmonic_num + 1L
  two_pi <- .yq_f32(2 * pi)
  amp32 <- .yq_f32(sine_amp)
  std32 <- .yq_f32(noise_std)
  out <- array(0, c(B, TT, H))
  for (b in seq_len(B)) {
    f0b <- f0_up[b, ]
    uv <- as.numeric(f0b > voiced_threshold)
    ph <- phase[b, , 1L]
    ph[1L] <- 0
    sw <- matrix(0, H, TT)
    for (k in seq_len(H)) {
      # F_mat row: f0 * k / sr with per-op float32 rounding, then the
      # double-accumulate/float32-out cumsum (torch CPU semantics),
      # fmod 1 (exact on float32 values), 2*pi scale, phase shift, sin.
      fk <- .yq_f32(.yq_f32(f0b * k) / sample_rate)
      cs <- .yq_f32(cumsum(fk))
      theta <- .yq_f32(two_pi * (cs %% 1))
      sw[k, ] <- .yq_f32(amp32 * sin(.yq_f32(theta + ph[k])))
    }
    # noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    namp <- .yq_f32(.yq_f32(uv * std32) +
      .yq_f32(.yq_f32((1 - uv) * amp32) / 3))
    noise <- .yq_f32(sweep(z[b, , , drop = FALSE][1L, , ], 2L, namp, "*"))
    sfin <- .yq_f32(.yq_f32(sweep(sw, 2L, uv, "*")) + noise)
    out[b, , ] <- t(sfin)
  }
  out
}

# NSF source module: harmonic sine bank -> linear merge -> tanh.
# f0_up is an R matrix [B, T_wav]; returns AnvlArray [B, 1, T_wav].
.yq_hifigan_source <- function(f0_up, w, phase, z) {
  sine <- .yq_hifigan_sine(f0_up, phase, z)
  sw <- anvl::nv_array(sine, dtype = "f32") # [B, T_wav, H+1]
  sm <- anvl::nv_tanh(yunque::linear(sw, w$l_linear_w, w$l_linear_b))
  anvl::nv_transpose(sm, c(1L, 3L, 2L)) # [B, 1, T_wav]
}

# Decode mel + source excitation into audio via the ISTFT head.
.yq_hifigan_decode <- function(mel, s, w) {
  win <- yunque::hann_window(16L)
  ss <- anvl::shape(s)
  sp <- yunque::stft(anvl::nv_reshape(s, c(ss[1L], ss[3L])),
    n_fft = 16L, hop_length = 4L, window = win, center = TRUE)
  s_stft <- anvl::nv_concatenate(sp$real, sp$imag, dimension = 2L)

  x <- yunque::conv1d(mel, w$conv_pre_w, w$conv_pre_b, padding = 3L)
  n_up <- length(w$ups)
  for (i in seq_len(n_up)) {
    x <- .yq_leaky_relu(x, 0.1)
    u <- w$ups[[i]]
    x <- yunque::conv_transpose1d(x, u$w, u$b, stride = u$stride,
      padding = u$padding)
    if (i == n_up) {
      x <- .yq_reflect_pad1d(x, 1L, 0L)
    }
    sd <- w$source_downs[[i]]
    si <- yunque::conv1d(s_stft, sd$w, sd$b, stride = sd$stride,
      padding = sd$padding)
    si <- .yq_hifigan_resblock(si, w$source_resblocks[[i]])
    xl <- anvl::shape(x)[3L]
    sl <- anvl::shape(si)[3L]
    if (sl > xl) {
      si <- .yq_slice3(si, 3L, 1L, xl)
    } else if (xl > sl) {
      x <- .yq_slice3(x, 3L, 1L, sl)
    }
    x <- x + si
    xs <- NULL
    for (j in seq_len(3L)) {
      r <- .yq_hifigan_resblock(x, w$resblocks[[(i - 1L) * 3L + j]])
      xs <- if (is.null(xs)) r else xs + r
    }
    x <- xs / 3
  }
  x <- .yq_leaky_relu(x, 0.01) # final lrelu uses the torch default slope
  x <- yunque::conv1d(x, w$conv_post_w, w$conv_post_b, padding = 3L)

  mag <- anvl::nv_min(anvl::nv_exp(.yq_slice3(x, 2L, 1L, 9L)), 100)
  p <- anvl::nv_sin(.yq_slice3(x, 2L, 10L, 18L))
  audio <- yunque::istft(mag * anvl::nv_cos(p), mag * anvl::nv_sin(p),
    n_fft = 16L, hop_length = 4L, window = win, center = TRUE)
  anvl::nv_clamp(min_val = -0.99, operand = audio, max_val = 0.99)
}

#' Load HiFT/HiFiGAN vocoder weights (anvl)
#'
#' Reconstructs the weight-normalized convolutions host-side
#' (\code{w = g * v / ||v||}, norm over all dims but the first --
#' torch's \code{weight_norm} with \code{dim = 0} for both Conv1d and
#' ConvTranspose1d) and bundles the S3Gen vocoder configuration
#' (upsample rates 8/5/3, source downsample 15/3/1, ISTFT 16/4).
#'
#' @param path Path to s3gen.safetensors.
#' @param prefix Key prefix (default \code{"mel2wav."}).
#' @return Nested list of AnvlArray weights for \code{\link{yq_hifigan}}.
#' @export
yq_hifigan_load_weights <- function(path, prefix = "mel2wav.") {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  rd <- function(k) {
    yunque::st_read(st, paste0(prefix, k), transpose = FALSE)
  }
  rdt <- function(k) {
    yunque::st_read(st, paste0(prefix, k), transpose = TRUE)
  }
  nv <- function(a) anvl::nv_array(a, dtype = "f32")
  wn <- function(base) {
    g <- rd(paste0(base, ".parametrizations.weight.original0"))
    v <- rd(paste0(base, ".parametrizations.weight.original1"))
    n <- sqrt(rowSums(matrix(v, nrow = dim(v)[1L])^2))
    nv(v * (c(g) / n))
  }
  bias <- function(base) nv(rd(paste0(base, ".bias")))
  alpha <- function(k) {
    a <- rd(k)
    nv(array(a, c(1L, length(a), 1L)))
  }
  resblock <- function(base, kernel) {
    branches <- lapply(1:3, function(j) {
      list(
        dilation = c(1L, 3L, 5L)[j],
        alpha1 = alpha(sprintf("%s.activations1.%d.alpha", base, j - 1L)),
        conv1_w = wn(sprintf("%s.convs1.%d", base, j - 1L)),
        conv1_b = bias(sprintf("%s.convs1.%d", base, j - 1L)),
        alpha2 = alpha(sprintf("%s.activations2.%d.alpha", base, j - 1L)),
        conv2_w = wn(sprintf("%s.convs2.%d", base, j - 1L)),
        conv2_b = bias(sprintf("%s.convs2.%d", base, j - 1L)))
    })
    list(kernel = as.integer(kernel), branches = branches)
  }
  up_rates <- c(8L, 5L, 3L)
  up_kernels <- c(16L, 11L, 7L)
  res_kernels <- c(3L, 7L, 11L)
  src_kernels <- c(7L, 7L, 11L)
  down_rates <- c(15L, 3L, 1L)
  list(
    f0_condnet = lapply(0:4, function(i) {
      list(w = wn(sprintf("f0_predictor.condnet.%d", i * 2L)),
        b = bias(sprintf("f0_predictor.condnet.%d", i * 2L)))
    }),
    f0_classifier_w = nv(rdt("f0_predictor.classifier.weight")),
    f0_classifier_b = nv(rd("f0_predictor.classifier.bias")),
    l_linear_w = nv(rdt("m_source.l_linear.weight")),
    l_linear_b = nv(rd("m_source.l_linear.bias")),
    conv_pre_w = wn("conv_pre"),
    conv_pre_b = bias("conv_pre"),
    ups = lapply(1:3, function(i) {
      list(w = wn(sprintf("ups.%d", i - 1L)),
        b = bias(sprintf("ups.%d", i - 1L)),
        stride = up_rates[i],
        padding = (up_kernels[i] - up_rates[i]) %/% 2L)
    }),
    source_downs = lapply(1:3, function(i) {
      u <- down_rates[i]
      list(w = nv(rd(sprintf("source_downs.%d.weight", i - 1L))),
        b = nv(rd(sprintf("source_downs.%d.bias", i - 1L))),
        stride = u,
        padding = if (u == 1L) 0L else u %/% 2L)
    }),
    source_resblocks = lapply(1:3, function(i) {
      resblock(sprintf("source_resblocks.%d", i - 1L), src_kernels[i])
    }),
    resblocks = lapply(1:9, function(idx) {
      resblock(sprintf("resblocks.%d", idx - 1L),
        res_kernels[(idx - 1L) %% 3L + 1L])
    }),
    conv_post_w = wn("conv_post"),
    conv_post_b = bias("conv_post"))
}

#' HiFT F0 predictor forward (anvl)
#'
#' Torch-free port of \code{conv_rnn_f0_predictor}: 5 ELU conv layers
#' plus an abs-linear classifier head.
#'
#' @param mel AnvlArray \code{[B, 80, T]} mel spectrogram.
#' @param w Weights from \code{\link{yq_hifigan_load_weights}}.
#'
#' @return AnvlArray \code{[B, T]} F0 in Hz.
#'
#' @export
yq_hifigan_f0 <- function(mel, w) {
  x <- mel
  for (cv in w$f0_condnet) {
    x <- yunque::elu(yunque::conv1d(x, cv$w, cv$b, padding = 1L))
  }
  x <- anvl::nv_transpose(x, c(1L, 3L, 2L)) # [B, T, C]
  x <- yunque::linear(x, w$f0_classifier_w, w$f0_classifier_b) # [B, T, 1]
  s <- anvl::shape(x)
  anvl::nv_abs(anvl::nv_reshape(x, c(s[1L], s[2L])))
}

#' HiFT/HiFiGAN vocoder forward (anvl)
#'
#' Torch-free port of \code{hift_generator$inference}: F0 prediction,
#' NSF harmonic-plus-noise source, upsampling stack with source STFT
#' fusion, and ISTFT synthesis. The two random draws of the reference
#' (per-harmonic initial phase, source noise) are explicit arguments;
#' NULL draws them from R's RNG.
#'
#' @param mel AnvlArray \code{[B, 80, T]} mel spectrogram.
#' @param w Weights from \code{\link{yq_hifigan_load_weights}}.
#' @param phase Optional R array \code{[B, 9, 1]} of initial harmonic
#'   phases in \code{(-pi, pi)} (the fundamental is zeroed internally).
#'   NULL draws them with \code{runif}.
#' @param noise Optional R array \code{[B, 9, T * 480]} of
#'   standard-normal source noise. NULL draws it with \code{rnorm}.
#' @param cache_source Optional AnvlArray or R array \code{[B, 1, L]}
#'   replacing the first \code{L} source samples (streaming continuity,
#'   as in the torch \code{inference}).
#'
#' @return List with \code{audio} (AnvlArray \code{[B, T * 480]}) and
#'   \code{source} (AnvlArray \code{[B, 1, T * 480]}).
#'
#' @export
yq_hifigan <- function(mel, w, phase = NULL, noise = NULL,
                       cache_source = NULL) {
  f0 <- yq_hifigan_f0(mel, w) # [B, T]
  f0_host <- as.array(f0)
  if (is.null(dim(f0_host))) {
    dim(f0_host) <- c(1L, length(f0_host))
  }
  B <- dim(f0_host)[1L]
  up <- 480L # prod(upsample_rates) * istft_hop_len
  T_wav <- dim(f0_host)[2L] * up
  H <- 9L # nb_harmonics + 1
  f0_up <- matrix(0, B, T_wav)
  for (b in seq_len(B)) {
    f0_up[b, ] <- rep(f0_host[b, ], each = up) # nearest upsample
  }
  if (is.null(phase)) {
    phase <- array(stats::runif(B * H, -pi, pi), c(B, H, 1L))
  }
  if (is.null(noise)) {
    noise <- array(stats::rnorm(B * H * T_wav), c(B, H, T_wav))
  }
  s <- .yq_hifigan_source(f0_up, w, phase, noise)
  if (!is.null(cache_source)) {
    cs <- if (is.numeric(cache_source)) {
      anvl::nv_array(cache_source, dtype = "f32")
    } else {
      cache_source
    }
    cl <- anvl::shape(cs)[3L]
    if (cl >= T_wav) {
      s <- .yq_slice3(cs, 3L, 1L, T_wav)
    } else if (cl > 0L) {
      s <- anvl::nv_concatenate(cs, .yq_slice3(s, 3L, cl + 1L, T_wav),
        dimension = 3L)
    }
  }
  audio <- .yq_hifigan_decode(mel, s, w)
  list(audio = audio, source = s)
}
