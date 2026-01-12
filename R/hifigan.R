# HiFiGAN Vocoder for chatteRbox
# Neural Source Filter + ISTFTNet for mel-to-waveform conversion

# ============================================================================
# Utility Functions
# ============================================================================

#' Get padding for convolution
#'
#' @param kernel_size Kernel size
#' @param dilation Dilation rate
#' @return Padding size
get_conv_padding <- function(kernel_size, dilation = 1) {
  as.integer((kernel_size * dilation - dilation) / 2)
}

#' Reflection padding for 1D (nn_reflection_pad1d equivalent)
#'
#' @param padding Integer vector c(left, right) for padding
#' @return nn_module
reflection_pad1d <- torch::nn_module(
  "ReflectionPad1d",

  initialize = function(padding) {
    if (length(padding) == 1) {
      self$padding <- c(padding, padding)
    } else {
      self$padding <- padding
    }
  },

  forward = function(x) {
    torch::nnf_pad(x, self$padding, mode = "reflect")
  }
)

# ============================================================================
# Snake Activation
# ============================================================================

#' Snake activation function
#'
#' Sine-based periodic activation: x + 1/a * sin^2(ax)
#' Reference: https://arxiv.org/abs/2006.08195
#'
#' @param in_features Number of input channels
#' @param alpha_trainable Whether alpha is trainable
#' @param alpha_logscale Whether to use log scale for alpha
#' @return nn_module
snake_activation <- torch::nn_module(
  "Snake",

  initialize = function(in_features, alpha_trainable = TRUE, alpha_logscale = FALSE) {
    self$in_features <- in_features
    self$alpha_logscale <- alpha_logscale
    self$no_div_by_zero <- 1e-9

    # Initialize alpha
    if (alpha_logscale) {
      self$alpha <- torch::nn_parameter(torch::torch_zeros(in_features))
    } else {
      self$alpha <- torch::nn_parameter(torch::torch_ones(in_features))
    }

    if (!alpha_trainable) {
      self$alpha$requires_grad <- FALSE
    }
  },

  forward = function(x) {
    # x: (B, C, T)
    alpha <- self$alpha$unsqueeze(1)$unsqueeze(3)  # (1, C, 1)

    if (self$alpha_logscale) {
      alpha <- torch::torch_exp(alpha)
    }

    # Snake: x + 1/a * sin^2(ax)
    x + (1.0 / (alpha + self$no_div_by_zero)) * torch::torch_pow(torch::torch_sin(x * alpha), 2)
  }
)

# ============================================================================
# Residual Block
# ============================================================================

#' HiFiGAN Residual Block
#'
#' @param channels Number of channels
#' @param kernel_size Kernel size
#' @param dilations List of dilation rates
#' @return nn_module
hifigan_resblock <- torch::nn_module(
  "HiFiGANResBlock",

  initialize = function(channels = 512, kernel_size = 3, dilations = c(1, 3, 5)) {
    self$convs1 <- torch::nn_module_list()
    self$convs2 <- torch::nn_module_list()
    self$activations1 <- torch::nn_module_list()
    self$activations2 <- torch::nn_module_list()

    for (dilation in dilations) {
      # First conv with dilation
      self$convs1$append(
        torch::nn_conv1d(channels, channels, kernel_size, stride = 1,
                         dilation = dilation,
                         padding = get_conv_padding(kernel_size, dilation))
      )
      self$activations1$append(snake_activation(channels))

      # Second conv without dilation
      self$convs2$append(
        torch::nn_conv1d(channels, channels, kernel_size, stride = 1,
                         dilation = 1,
                         padding = get_conv_padding(kernel_size, 1))
      )
      self$activations2$append(snake_activation(channels))
    }
  },

  forward = function(x) {
    for (i in seq_along(self$convs1)) {
      xt <- self$activations1[[i]](x)
      xt <- self$convs1[[i]](xt)
      xt <- self$activations2[[i]](xt)
      xt <- self$convs2[[i]](xt)
      x <- xt + x  # Residual
    }
    x
  }
)

# ============================================================================
# F0 Predictor
# ============================================================================

#' Convolutional RNN F0 Predictor
#'
#' @param in_channels Input channels (mel bins)
#' @param cond_channels Hidden channels
#' @return nn_module
conv_rnn_f0_predictor <- torch::nn_module(
  "ConvRNNF0Predictor",

  initialize = function(in_channels = 80, cond_channels = 512) {
    # 5-layer convnet
    self$condnet <- torch::nn_sequential(
      torch::nn_conv1d(in_channels, cond_channels, kernel_size = 3, padding = 1),
      torch::nn_elu(),
      torch::nn_conv1d(cond_channels, cond_channels, kernel_size = 3, padding = 1),
      torch::nn_elu(),
      torch::nn_conv1d(cond_channels, cond_channels, kernel_size = 3, padding = 1),
      torch::nn_elu(),
      torch::nn_conv1d(cond_channels, cond_channels, kernel_size = 3, padding = 1),
      torch::nn_elu(),
      torch::nn_conv1d(cond_channels, cond_channels, kernel_size = 3, padding = 1),
      torch::nn_elu()
    )

    # Output classifier
    self$classifier <- torch::nn_linear(cond_channels, 1)
  },

  forward = function(x) {
    # x: (B, mel_bins, T)
    x <- self$condnet(x)
    x <- x$transpose(2, 3)  # (B, T, C)
    torch::torch_abs(self$classifier(x)$squeeze(3))  # (B, T)
  }
)

# ============================================================================
# Sine Generator for Neural Source Filter
# ============================================================================

#' Sine Generator
#'
#' Generates sine waveforms from F0 for source-filter synthesis
#'
#' @param sample_rate Sampling rate in Hz
#' @param harmonic_num Number of harmonics
#' @param sine_amp Sine amplitude
#' @param noise_std Noise standard deviation
#' @param voiced_threshold F0 threshold for voiced/unvoiced
#' @return nn_module
sine_gen <- torch::nn_module(
  "SineGen",

  initialize = function(sample_rate, harmonic_num = 0, sine_amp = 0.1,
                        noise_std = 0.003, voiced_threshold = 0) {
    self$sine_amp <- sine_amp
    self$noise_std <- noise_std
    self$harmonic_num <- harmonic_num
    self$sampling_rate <- sample_rate
    self$voiced_threshold <- voiced_threshold
  },

  .f02uv = function(f0) {
    # Generate voiced/unvoiced signal
    (f0 > self$voiced_threshold)$to(dtype = torch::torch_float32())
  },

  forward = function(f0) {
    # f0: (B, 1, T) in Hz

    torch::with_no_grad({
      batch_size <- f0$size(1)
      seq_len <- f0$size(3)
      device <- f0$device

      # Frequency matrix for all harmonics
      F_mat <- torch::torch_zeros(c(batch_size, self$harmonic_num + 1, seq_len),
                                   device = device)

      for (i in 0:self$harmonic_num) {
        F_mat[, i + 1, ] <- f0$squeeze(2) * (i + 1) / self$sampling_rate
      }

      # Phase accumulation (cumsum)
      theta_mat <- 2 * pi * (torch::torch_cumsum(F_mat, dim = 3) %% 1)

      # Random initial phase (except fundamental)
      phase_vec <- torch::torch_empty(c(batch_size, self$harmonic_num + 1, 1),
                                       device = device)$uniform_(-pi, pi)
      phase_vec[, 1, ] <- 0  # Fundamental starts at 0

      # Generate sine waves
      sine_waves <- self$sine_amp * torch::torch_sin(theta_mat + phase_vec)

      # Voiced/unvoiced mask
      uv <- self$.f02uv(f0)

      # Noise amplitude: voiced = noise_std, unvoiced = sine_amp/3
      noise_amp <- uv * self$noise_std + (1 - uv) * self$sine_amp / 3
      noise <- noise_amp * torch::torch_randn_like(sine_waves)

      # Zero unvoiced regions and add noise
      sine_waves <- sine_waves * uv + noise
    })

    list(sine_waves = sine_waves, uv = uv, noise = noise)
  }
)

# ============================================================================
# Source Module (Harmonic + Noise Source Filter)
# ============================================================================

#' Source Module for Neural Source Filter
#'
#' @param sample_rate Sampling rate
#' @param upsample_scale Upsampling factor
#' @param harmonic_num Number of harmonics
#' @param sine_amp Sine amplitude
#' @param add_noise_std Noise std
#' @param voiced_threshold Voiced threshold
#' @return nn_module
source_module_hn_nsf <- torch::nn_module(
  "SourceModuleHnNSF",

  initialize = function(sample_rate, upsample_scale, harmonic_num = 0,
                        sine_amp = 0.1, add_noise_std = 0.003, voiced_threshold = 0) {
    self$sine_amp <- sine_amp
    self$noise_std <- add_noise_std

    # Sine generator
    self$l_sin_gen <- sine_gen(sample_rate, harmonic_num, sine_amp,
                               add_noise_std, voiced_threshold)

    # Linear combination of harmonics
    self$l_linear <- torch::nn_linear(harmonic_num + 1, 1)
    self$l_tanh <- torch::nn_tanh()
  },

  forward = function(x) {
    # x: (B, T, 1) F0 in Hz

    # Generate sine source
    result <- self$l_sin_gen(x$transpose(2, 3))
    sine_wavs <- result$sine_waves$transpose(2, 3)  # (B, T, H+1)
    uv <- result$uv$transpose(2, 3)

    # Merge harmonics
    sine_merge <- self$l_tanh(self$l_linear(sine_wavs))  # (B, T, 1)

    # Noise source
    noise <- torch::torch_randn_like(uv) * self$sine_amp / 3

    list(sine_merge = sine_merge, noise = noise, uv = uv)
  }
)

# ============================================================================
# HiFTGenerator (Main Vocoder)
# ============================================================================

#' HiFTNet Generator
#'
#' Neural Source Filter + ISTFTNet
#' Reference: https://arxiv.org/abs/2309.09493
#'
#' @param in_channels Input mel channels
#' @param base_channels Base channel count
#' @param nb_harmonics Number of harmonics for source filter
#' @param sampling_rate Output sample rate
#' @param nsf_alpha NSF sine amplitude
#' @param nsf_sigma NSF noise std
#' @param nsf_voiced_threshold F0 voiced threshold
#' @param upsample_rates Upsampling rates
#' @param upsample_kernel_sizes Upsampling kernel sizes
#' @param istft_n_fft ISTFT FFT size
#' @param istft_hop_len ISTFT hop length
#' @param resblock_kernel_sizes ResBlock kernel sizes
#' @param resblock_dilation_sizes ResBlock dilations
#' @param source_resblock_kernel_sizes Source resblock kernels
#' @param source_resblock_dilation_sizes Source resblock dilations
#' @param lrelu_slope LeakyReLU slope
#' @param audio_limit Output clipping limit
#' @return nn_module
hift_generator <- torch::nn_module(
  "HiFTGenerator",

  initialize = function(in_channels = 80,
                        base_channels = 512,
                        nb_harmonics = 8,
                        sampling_rate = 22050,
                        nsf_alpha = 0.1,
                        nsf_sigma = 0.003,
                        nsf_voiced_threshold = 10,
                        upsample_rates = c(8, 8),
                        upsample_kernel_sizes = c(16, 16),
                        istft_n_fft = 16,
                        istft_hop_len = 4,
                        resblock_kernel_sizes = c(3, 7, 11),
                        resblock_dilation_sizes = list(c(1, 3, 5), c(1, 3, 5), c(1, 3, 5)),
                        source_resblock_kernel_sizes = c(7, 11),
                        source_resblock_dilation_sizes = list(c(1, 3, 5), c(1, 3, 5)),
                        lrelu_slope = 0.1,
                        audio_limit = 0.99) {

    self$out_channels <- 1
    self$nb_harmonics <- nb_harmonics
    self$sampling_rate <- sampling_rate
    self$istft_n_fft <- istft_n_fft
    self$istft_hop_len <- istft_hop_len
    self$lrelu_slope <- lrelu_slope
    self$audio_limit <- audio_limit
    self$num_kernels <- length(resblock_kernel_sizes)
    self$num_upsamples <- length(upsample_rates)

    # Total upsampling factor
    total_upsample <- prod(upsample_rates) * istft_hop_len

    # F0 predictor
    self$f0_predictor <- conv_rnn_f0_predictor(in_channels, 512)

    # Source module for excitation signal
    self$m_source <- source_module_hn_nsf(
      sample_rate = sampling_rate,
      upsample_scale = total_upsample,
      harmonic_num = nb_harmonics,
      sine_amp = nsf_alpha,
      add_noise_std = nsf_sigma,
      voiced_threshold = nsf_voiced_threshold
    )

    # F0 upsampling
    self$f0_upsamp <- torch::nn_upsample(scale_factor = total_upsample)

    # Initial convolution
    self$conv_pre <- torch::nn_conv1d(in_channels, base_channels, 7, stride = 1, padding = 3)

    # Upsampling layers
    self$ups <- torch::nn_module_list()
    for (i in seq_along(upsample_rates)) {
      u <- upsample_rates[i]
      k <- upsample_kernel_sizes[i]
      in_ch <- base_channels %/% (2^(i - 1))
      out_ch <- base_channels %/% (2^i)

      self$ups$append(
        torch::nn_conv_transpose1d(in_ch, out_ch, k, stride = u,
                                    padding = (k - u) %/% 2)
      )
    }

    # Source downsampling and resblocks
    self$source_downs <- torch::nn_module_list()
    self$source_resblocks <- torch::nn_module_list()

    # Compute downsample rates
    downsample_rates <- c(1, rev(upsample_rates[-length(upsample_rates)]))
    downsample_cum <- cumprod(downsample_rates)

    for (i in seq_along(source_resblock_kernel_sizes)) {
      u <- rev(downsample_cum)[i]
      k <- source_resblock_kernel_sizes[i]
      d <- source_resblock_dilation_sizes[[i]]
      out_ch <- base_channels %/% (2^i)

      if (u == 1) {
        self$source_downs$append(
          torch::nn_conv1d(istft_n_fft + 2, out_ch, 1, stride = 1)
        )
      } else {
        self$source_downs$append(
          torch::nn_conv1d(istft_n_fft + 2, out_ch, u * 2, stride = u, padding = u %/% 2)
        )
      }

      self$source_resblocks$append(hifigan_resblock(out_ch, k, d))
    }

    # Main resblocks
    self$resblocks <- torch::nn_module_list()
    for (i in seq_along(self$ups)) {
      ch <- base_channels %/% (2^i)
      for (j in seq_along(resblock_kernel_sizes)) {
        k <- resblock_kernel_sizes[j]
        d <- resblock_dilation_sizes[[j]]
        self$resblocks$append(hifigan_resblock(ch, k, d))
      }
    }

    # Output convolution
    final_ch <- base_channels %/% (2^length(upsample_rates))
    self$conv_post <- torch::nn_conv1d(final_ch, istft_n_fft + 2, 7, stride = 1, padding = 3)

    # Reflection padding for alignment
    self$reflection_pad <- reflection_pad1d(c(1, 0))

    # STFT window (Hann)
    hann_window <- 0.5 * (1 - cos(2 * pi * (0:(istft_n_fft - 1)) / istft_n_fft))
    self$stft_window <- torch::nn_buffer(
      torch::torch_tensor(hann_window, dtype = torch::torch_float32())
    )
  },

  .stft = function(x) {
    # x: (B, T) waveform
    spec <- torch::torch_stft(
      x,
      n_fft = self$istft_n_fft,
      hop_length = self$istft_hop_len,
      win_length = self$istft_n_fft,
      window = self$stft_window$to(device = x$device),
      return_complex = TRUE
    )

    # Convert to real/imag
    spec_real <- torch::torch_real(spec)
    spec_imag <- torch::torch_imag(spec)

    list(real = spec_real, imag = spec_imag)
  },

  .istft = function(magnitude, phase) {
    # Clip magnitude
    magnitude <- torch::torch_clamp(magnitude, max = 100)

    # Convert to complex
    real <- magnitude * torch::torch_cos(phase)
    imag <- magnitude * torch::torch_sin(phase)
    complex_spec <- torch::torch_complex(real, imag)

    # Inverse STFT
    torch::torch_istft(
      complex_spec,
      n_fft = self$istft_n_fft,
      hop_length = self$istft_hop_len,
      win_length = self$istft_n_fft,
      window = self$stft_window$to(device = magnitude$device)
    )
  },

  decode = function(x, s = NULL) {
    # x: (B, mel, T) mel spectrogram
    # s: (B, 1, T_wav) source signal

    device <- x$device

    if (is.null(s)) {
      s <- torch::torch_zeros(c(x$size(1), 1, 0), device = device)
    }

    # STFT of source
    if (s$size(3) > 0) {
      stft_result <- self$.stft(s$squeeze(2))
      s_stft <- torch::torch_cat(list(stft_result$real, stft_result$imag), dim = 2)
    } else {
      # Empty source - create zero tensor
      s_stft <- torch::torch_zeros(c(x$size(1), self$istft_n_fft + 2, 1), device = device)
    }

    # Initial conv
    x <- self$conv_pre(x)

    # Upsampling with source fusion
    for (i in seq_len(self$num_upsamples)) {
      x <- torch::nnf_leaky_relu(x, self$lrelu_slope)
      x <- self$ups[[i]](x)

      # Reflection pad for last upsample
      if (i == self$num_upsamples) {
        x <- self$reflection_pad(x)
      }

      # Source fusion
      si <- self$source_downs[[i]](s_stft)
      si <- self$source_resblocks[[i]](si)
      x <- x + si

      # ResBlocks
      xs <- NULL
      for (j in seq_len(self$num_kernels)) {
        idx <- (i - 1) * self$num_kernels + j
        if (is.null(xs)) {
          xs <- self$resblocks[[idx]](x)
        } else {
          xs <- xs + self$resblocks[[idx]](x)
        }
      }
      x <- xs / self$num_kernels
    }

    # Output
    x <- torch::nnf_leaky_relu(x)
    x <- self$conv_post(x)

    # Split magnitude and phase
    n_freq <- self$istft_n_fft %/% 2 + 1
    magnitude <- torch::torch_exp(x[, 1:n_freq, ])
    phase <- torch::torch_sin(x[, (n_freq + 1):(self$istft_n_fft + 2), ])

    # ISTFT synthesis
    audio <- self$.istft(magnitude, phase)

    # Clip output
    torch::torch_clamp(audio, -self$audio_limit, self$audio_limit)
  },

  forward = function(speech_feat) {
    # speech_feat: (B, T, mel) -> transpose to (B, mel, T)
    speech_feat <- speech_feat$transpose(2, 3)

    # Predict F0 from mel
    f0 <- self$f0_predictor(speech_feat)  # (B, T)

    # Upsample F0 to sample rate
    s <- self$f0_upsamp(f0$unsqueeze(2))$transpose(2, 3)  # (B, T_wav, 1)

    # Generate source signal
    source_result <- self$m_source(s)
    s <- source_result$sine_merge$transpose(2, 3)  # (B, 1, T_wav)

    # Decode mel + source -> audio
    generated_speech <- self$decode(speech_feat, s)

    list(audio = generated_speech, f0 = f0)
  },

  inference = function(speech_feat, cache_source = NULL) {
    # speech_feat: (B, mel, T)
    device <- speech_feat$device

    # Predict F0
    f0 <- self$f0_predictor(speech_feat)

    # Upsample F0
    s <- self$f0_upsamp(f0$unsqueeze(2))$transpose(2, 3)

    # Generate source
    source_result <- self$m_source(s)
    s <- source_result$sine_merge$transpose(2, 3)

    # Use cached source to avoid glitches at boundaries
    if (!is.null(cache_source) && cache_source$size(3) > 0) {
      cache_len <- cache_source$size(3)
      s[, , 1:cache_len] <- cache_source
    }

    # Decode
    generated_speech <- self$decode(speech_feat, s)

    list(audio = generated_speech, source = s)
  }
)

# ============================================================================
# Weight Loading
# ============================================================================

#' Load HiFiGAN weights from state dictionary
#'
#' @param model HiFTGenerator model
#' @param state_dict Named list of tensors
#' @param prefix Prefix for weight keys (default "mel2wav.")
#' @return Model with loaded weights
#' @export
load_hifigan_weights <- function(model, state_dict, prefix = "mel2wav.") {
  # Map weights from Python model to R implementation
  # This requires careful attention to naming conventions

  # Helper to copy weight if exists
  copy_if_exists <- function(r_param, key) {
    full_key <- paste0(prefix, key)
    if (full_key %in% names(state_dict)) {
      r_param$copy_(state_dict[[full_key]])
      return(TRUE)
    }
    FALSE
  }

  # F0 predictor
  f0_keys <- grep(paste0("^", prefix, "f0_predictor\\."), names(state_dict), value = TRUE)
  if (length(f0_keys) > 0) {
    # condnet layers (5 conv layers)
    for (i in 0:4) {
      conv_key <- sprintf("f0_predictor.condnet.%d.weight", i * 2)
      if (paste0(prefix, conv_key) %in% names(state_dict)) {
        # Sequential index: conv at 0, 2, 4, 6, 8
        model$f0_predictor$condnet[[i * 2 + 1]]$weight$copy_(
          state_dict[[paste0(prefix, conv_key)]]
        )
      }
      bias_key <- sprintf("f0_predictor.condnet.%d.bias", i * 2)
      if (paste0(prefix, bias_key) %in% names(state_dict)) {
        model$f0_predictor$condnet[[i * 2 + 1]]$bias$copy_(
          state_dict[[paste0(prefix, bias_key)]]
        )
      }
    }
    # classifier
    copy_if_exists(model$f0_predictor$classifier$weight, "f0_predictor.classifier.weight")
    copy_if_exists(model$f0_predictor$classifier$bias, "f0_predictor.classifier.bias")
  }

  # Source module
  copy_if_exists(model$m_source$l_linear$weight, "m_source.l_linear.weight")
  copy_if_exists(model$m_source$l_linear$bias, "m_source.l_linear.bias")

  # conv_pre
  copy_if_exists(model$conv_pre$weight, "conv_pre.weight")
  copy_if_exists(model$conv_pre$bias, "conv_pre.bias")

  # Upsampling layers
  for (i in seq_along(model$ups)) {
    key_w <- sprintf("ups.%d.weight", i - 1)
    key_b <- sprintf("ups.%d.bias", i - 1)
    copy_if_exists(model$ups[[i]]$weight, key_w)
    copy_if_exists(model$ups[[i]]$bias, key_b)
  }

  # Source downs and resblocks
  for (i in seq_along(model$source_downs)) {
    key_w <- sprintf("source_downs.%d.weight", i - 1)
    key_b <- sprintf("source_downs.%d.bias", i - 1)
    copy_if_exists(model$source_downs[[i]]$weight, key_w)
    copy_if_exists(model$source_downs[[i]]$bias, key_b)
  }

  # Main resblocks - complex nested structure
  for (i in seq_along(model$resblocks)) {
    block_prefix <- sprintf("resblocks.%d.", i - 1)
    # Each resblock has convs1, convs2, activations1, activations2
    for (j in seq_along(model$resblocks[[i]]$convs1)) {
      conv1_w <- paste0(block_prefix, sprintf("convs1.%d.weight", j - 1))
      conv1_b <- paste0(block_prefix, sprintf("convs1.%d.bias", j - 1))
      conv2_w <- paste0(block_prefix, sprintf("convs2.%d.weight", j - 1))
      conv2_b <- paste0(block_prefix, sprintf("convs2.%d.bias", j - 1))

      copy_if_exists(model$resblocks[[i]]$convs1[[j]]$weight, conv1_w)
      copy_if_exists(model$resblocks[[i]]$convs1[[j]]$bias, conv1_b)
      copy_if_exists(model$resblocks[[i]]$convs2[[j]]$weight, conv2_w)
      copy_if_exists(model$resblocks[[i]]$convs2[[j]]$bias, conv2_b)

      # Snake activations
      act1_key <- paste0(block_prefix, sprintf("activations1.%d.alpha", j - 1))
      act2_key <- paste0(block_prefix, sprintf("activations2.%d.alpha", j - 1))
      copy_if_exists(model$resblocks[[i]]$activations1[[j]]$alpha, act1_key)
      copy_if_exists(model$resblocks[[i]]$activations2[[j]]$alpha, act2_key)
    }
  }

  # conv_post
  copy_if_exists(model$conv_post$weight, "conv_post.weight")
  copy_if_exists(model$conv_post$bias, "conv_post.bias")

  model
}

# ============================================================================
# Factory Function
# ============================================================================

#' Create HiFiGAN vocoder with S3Gen configuration
#'
#' @param device Target device
#' @return HiFTGenerator module
#' @export
create_s3gen_vocoder <- function(device = "cpu") {
  # S3Gen uses specific HiFiGAN configuration
  model <- hift_generator(
    in_channels = 80,
    base_channels = 512,
    nb_harmonics = 8,
    sampling_rate = 24000,
    nsf_alpha = 0.1,
    nsf_sigma = 0.003,
    nsf_voiced_threshold = 10,
    upsample_rates = c(8, 5, 3),  # 8*5*3 = 120, with hop_len 4 gives ~480x
    upsample_kernel_sizes = c(16, 10, 6),
    istft_n_fft = 16,
    istft_hop_len = 4,
    resblock_kernel_sizes = c(3, 7, 11),
    resblock_dilation_sizes = list(c(1, 3, 5), c(1, 3, 5), c(1, 3, 5)),
    source_resblock_kernel_sizes = c(7, 11, 3),
    source_resblock_dilation_sizes = list(c(1, 3, 5), c(1, 3, 5), c(1, 3, 5)),
    lrelu_slope = 0.1,
    audio_limit = 0.99
  )

  model$to(device = device)
}
