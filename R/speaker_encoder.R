# CAMPPlus Speaker Encoder (xvector) for chatteRbox
# DenseNet-style TDNN for speaker embeddings (192-dim)

# ============================================================================
# Helper Functions
# ============================================================================

#' Statistics pooling
#'
#' @param x Input tensor (batch, channels, time)
#' @return Statistics tensor (batch, channels * 2)
statistics_pooling <- function(x) {
  mean_x <- torch::torch_mean(x, dim = 3)
  std_x <- torch::torch_std(x, dim = 3, unbiased = TRUE)
  torch::torch_cat(list(mean_x, std_x), dim = 2)
}

# ============================================================================
# Basic Blocks
# ============================================================================

#' Basic residual block for FCM
#'
#' @param in_planes Input channels
#' @param planes Output channels
#' @param stride Stride for downsampling
#' @return nn_module
basic_res_block <- torch::nn_module(
  "BasicResBlock",

  initialize = function(in_planes, planes, stride = 1) {
    self$conv1 <- torch::nn_conv2d(in_planes, planes, kernel_size = 3,
                                    stride = c(stride, 1), padding = 1, bias = FALSE)
    self$bn1 <- torch::nn_batch_norm2d(planes)
    self$conv2 <- torch::nn_conv2d(planes, planes, kernel_size = 3,
                                    stride = 1, padding = 1, bias = FALSE)
    self$bn2 <- torch::nn_batch_norm2d(planes)

    # Shortcut connection
    if (stride != 1 || in_planes != planes) {
      self$shortcut <- torch::nn_sequential(
        torch::nn_conv2d(in_planes, planes, kernel_size = 1,
                         stride = c(stride, 1), bias = FALSE),
        torch::nn_batch_norm2d(planes)
      )
    } else {
      self$shortcut <- NULL
    }
  },

  forward = function(x) {
    out <- torch::nnf_relu(self$bn1(self$conv1(x)))
    out <- self$bn2(self$conv2(out))

    if (!is.null(self$shortcut)) {
      out <- out + self$shortcut(x)
    } else {
      out <- out + x
    }

    torch::nnf_relu(out)
  }
)

#' Factorized Convolutional Module (FCM)
#'
#' @param m_channels Number of channels
#' @param feat_dim Input feature dimension (mel bins)
#' @return nn_module
fcm_module <- torch::nn_module(
  "FCM",

  initialize = function(m_channels = 32, feat_dim = 80) {
    self$in_planes <- m_channels

    self$conv1 <- torch::nn_conv2d(1, m_channels, kernel_size = 3,
                                    stride = 1, padding = 1, bias = FALSE)
    self$bn1 <- torch::nn_batch_norm2d(m_channels)

    # Two residual layers
    self$layer1 <- self$.make_layer(m_channels, 2, stride = 2)
    self$layer2 <- self$.make_layer(m_channels, 2, stride = 2)

    self$conv2 <- torch::nn_conv2d(m_channels, m_channels, kernel_size = 3,
                                    stride = c(2, 1), padding = 1, bias = FALSE)
    self$bn2 <- torch::nn_batch_norm2d(m_channels)

    self$out_channels <- m_channels * (feat_dim %/% 8)
  },

  .make_layer = function(planes, num_blocks, stride) {
    layers <- list()
    strides <- c(stride, rep(1, num_blocks - 1))

    for (s in strides) {
      layers[[length(layers) + 1]] <- basic_res_block(self$in_planes, planes, s)
      self$in_planes <- planes
    }

    torch::nn_sequential(!!!layers)
  },

  forward = function(x) {
    x <- x$unsqueeze(2)  # Add channel dim
    out <- torch::nnf_relu(self$bn1(self$conv1(x)))
    out <- self$layer1(out)
    out <- self$layer2(out)
    out <- torch::nnf_relu(self$bn2(self$conv2(out)))

    # Reshape: (B, C, H, W) -> (B, C*H, W)
    shape <- out$shape
    out$reshape(c(shape[1], shape[2] * shape[3], shape[4]))
  }
)

#' TDNN Layer
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param kernel_size Kernel size
#' @param stride Stride
#' @param dilation Dilation
#' @return nn_module
tdnn_layer <- torch::nn_module(
  "TDNNLayer",

  initialize = function(in_channels, out_channels, kernel_size,
                        stride = 1, dilation = 1, padding = NULL) {
    if (is.null(padding)) {
      padding <- ((kernel_size - 1) %/% 2) * dilation
    }

    self$conv <- torch::nn_conv1d(in_channels, out_channels, kernel_size,
                                   stride = stride, padding = padding,
                                   dilation = dilation, bias = FALSE)
    self$bn <- torch::nn_batch_norm1d(out_channels)
  },

  forward = function(x) {
    torch::nnf_relu(self$bn(self$conv(x)))
  }
)

#' CAM (Context-Aware Masking) Layer
#'
#' @param bn_channels Bottleneck channels
#' @param out_channels Output channels
#' @param kernel_size Kernel size
#' @param stride Stride
#' @param padding Padding
#' @param dilation Dilation
#' @param reduction Channel reduction factor
#' @return nn_module
cam_layer <- torch::nn_module(
  "CAMLayer",

  initialize = function(bn_channels, out_channels, kernel_size,
                        stride = 1, padding = 0, dilation = 1, reduction = 2) {
    self$linear_local <- torch::nn_conv1d(bn_channels, out_channels, kernel_size,
                                           stride = stride, padding = padding,
                                           dilation = dilation, bias = FALSE)
    self$linear1 <- torch::nn_conv1d(bn_channels, bn_channels %/% reduction, 1)
    self$relu <- torch::nn_relu()
    self$linear2 <- torch::nn_conv1d(bn_channels %/% reduction, out_channels, 1)
    self$sigmoid <- torch::nn_sigmoid()
  },

  .seg_pooling = function(x, seg_len = 100) {
    # Segment-based average pooling
    pooled <- torch::nnf_avg_pool1d(x, kernel_size = seg_len, stride = seg_len,
                                     ceil_mode = TRUE)

    # Expand back to original length
    shape <- pooled$shape
    expanded <- pooled$unsqueeze(4)$expand(c(shape[1], shape[2], shape[3], seg_len))
    expanded <- expanded$reshape(c(shape[1], shape[2], -1))

    # Trim to match input length
    expanded[, , 1:x$size(3)]
  },

  forward = function(x) {
    y <- self$linear_local(x)

    # Global context + segment context
    context <- x$mean(dim = 3, keepdim = TRUE) + self$.seg_pooling(x)
    context <- self$relu(self$linear1(context))
    m <- self$sigmoid(self$linear2(context))

    y * m
  }
)

#' CAM Dense TDNN Layer
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param bn_channels Bottleneck channels
#' @param kernel_size Kernel size
#' @param dilation Dilation
#' @return nn_module
cam_dense_tdnn_layer <- torch::nn_module(
  "CAMDenseTDNNLayer",

  initialize = function(in_channels, out_channels, bn_channels,
                        kernel_size, dilation = 1) {
    padding <- ((kernel_size - 1) %/% 2) * dilation

    self$bn1 <- torch::nn_batch_norm1d(in_channels)
    self$linear1 <- torch::nn_conv1d(in_channels, bn_channels, 1, bias = FALSE)
    self$bn2 <- torch::nn_batch_norm1d(bn_channels)
    self$cam <- cam_layer(bn_channels, out_channels, kernel_size,
                          padding = padding, dilation = dilation)
  },

  forward = function(x) {
    out <- torch::nnf_relu(self$bn1(x))
    out <- self$linear1(out)
    out <- torch::nnf_relu(self$bn2(out))
    self$cam(out)
  }
)

#' CAM Dense TDNN Block (multiple layers with dense connections)
#'
#' @param num_layers Number of layers
#' @param in_channels Input channels
#' @param out_channels Output channels per layer
#' @param bn_channels Bottleneck channels
#' @param kernel_size Kernel size
#' @param dilation Dilation
#' @return nn_module
cam_dense_tdnn_block <- torch::nn_module(
  "CAMDenseTDNNBlock",

  initialize = function(num_layers, in_channels, out_channels,
                        bn_channels, kernel_size, dilation = 1) {
    self$layers <- torch::nn_module_list()

    for (i in seq_len(num_layers)) {
      layer_in <- in_channels + (i - 1) * out_channels
      self$layers$append(
        cam_dense_tdnn_layer(layer_in, out_channels, bn_channels, kernel_size, dilation)
      )
    }
  },

  forward = function(x) {
    for (layer in self$layers) {
      out <- layer(x)
      x <- torch::torch_cat(list(x, out), dim = 2)
    }
    x
  }
)

#' Transit layer (channel reduction)
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param bias Whether to use bias (default FALSE)
#' @return nn_module
transit_layer <- torch::nn_module(
  "TransitLayer",

  initialize = function(in_channels, out_channels, bias = FALSE) {
    self$bn <- torch::nn_batch_norm1d(in_channels)
    self$conv <- torch::nn_conv1d(in_channels, out_channels, 1, bias = bias)
  },

  forward = function(x) {
    self$conv(torch::nnf_relu(self$bn(x)))
  }
)

#' Dense layer for final embedding
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @return nn_module
dense_layer <- torch::nn_module(
  "DenseLayer",

  initialize = function(in_channels, out_channels) {
    self$conv <- torch::nn_conv1d(in_channels, out_channels, 1, bias = FALSE)
    self$bn <- torch::nn_batch_norm1d(out_channels, affine = FALSE)
  },

  forward = function(x) {
    if (x$dim() == 2) {
      x <- x$unsqueeze(3)
      x <- self$bn(self$conv(x))$squeeze(3)
    } else {
      x <- self$bn(self$conv(x))
    }
    x
  }
)

# ============================================================================
# CAMPPlus Model
# ============================================================================

#' CAMPPlus speaker encoder
#'
#' @param feat_dim Input feature dimension (default 80)
#' @param embedding_size Output embedding size (default 192)
#' @param growth_rate Dense block growth rate (default 32)
#' @param init_channels Initial TDNN channels (default 128)
#' @return nn_module
campplus <- torch::nn_module(
  "CAMPPlus",

  initialize = function(feat_dim = 80, embedding_size = 192,
                        growth_rate = 32, init_channels = 128) {
    self$head <- fcm_module(m_channels = 32, feat_dim = feat_dim)

    channels <- self$head$out_channels

    # Initial TDNN
    self$tdnn <- tdnn_layer(channels, init_channels, kernel_size = 5,
                            stride = 2, dilation = 1)
    channels <- init_channels

    # Three dense blocks with transit layers
    # Block 1: 12 layers, kernel=3, dilation=1
    self$block1 <- cam_dense_tdnn_block(12, channels, growth_rate,
                                         growth_rate * 4, 3, dilation = 1)
    channels <- channels + 12 * growth_rate
    self$transit1 <- transit_layer(channels, channels %/% 2)
    channels <- channels %/% 2

    # Block 2: 24 layers, kernel=3, dilation=2
    self$block2 <- cam_dense_tdnn_block(24, channels, growth_rate,
                                         growth_rate * 4, 3, dilation = 2)
    channels <- channels + 24 * growth_rate
    self$transit2 <- transit_layer(channels, channels %/% 2)
    channels <- channels %/% 2

    # Block 3: 16 layers, kernel=3, dilation=2
    self$block3 <- cam_dense_tdnn_block(16, channels, growth_rate,
                                         growth_rate * 4, 3, dilation = 2)
    channels <- channels + 16 * growth_rate
    self$transit3 <- transit_layer(channels, channels %/% 2)
    channels <- channels %/% 2

    # Output layers
    self$out_bn <- torch::nn_batch_norm1d(channels)
    self$final_channels <- channels

    # Final dense layer (after stats pooling)
    self$dense <- dense_layer(channels * 2, embedding_size)
  },

  forward = function(x) {
    # x: (B, T, F) -> (B, F, T)
    x <- x$permute(c(1, 3, 2))

    # FCM head
    x <- self$head(x)

    # TDNN
    x <- self$tdnn(x)

    # Dense blocks with transit
    x <- self$block1(x)
    x <- self$transit1(x)

    x <- self$block2(x)
    x <- self$transit2(x)

    x <- self$block3(x)
    x <- self$transit3(x)

    # Output norm
    x <- torch::nnf_relu(self$out_bn(x))

    # Statistics pooling
    x <- statistics_pooling(x)

    # Final embedding
    self$dense(x)
  },

  inference = function(audio) {
    # Compute fbank features (80 mel bins)
    # This would normally use torchaudio kaldi features
    # For now, we use our mel spectrogram computation

    self(audio$to(dtype = torch::torch_float32()))
  }
)

# ============================================================================
# High-level Functions
# ============================================================================

#' Compute xvector speaker embedding from audio
#'
#' @param model CAMPPlus model
#' @param audio Audio samples (tensor or numeric)
#' @param sr Sample rate
#' @return Speaker embedding (1, 192)
#' @export
compute_xvector_embedding <- function(model, audio, sr) {
  device <- next(model$parameters())$device

  # Convert to tensor if needed
  if (!inherits(audio, "torch_tensor")) {
    audio <- torch::torch_tensor(audio, dtype = torch::torch_float32())
  }

  # Ensure 2D (batch, time)
  if (audio$dim() == 1) {
    audio <- audio$unsqueeze(1)
  }

  # Resample to 16kHz if needed
  target_sr <- 16000
  if (sr != target_sr) {
    audio_np <- as.numeric(audio$cpu())
    audio_np <- resample_audio(audio_np, sr, target_sr)
    audio <- torch::torch_tensor(audio_np, dtype = torch::torch_float32())$unsqueeze(1)
  }

  # Compute fbank features (80 mel bins, mimicking Kaldi fbank)
  mel <- compute_mel_spectrogram(
    audio,
    n_fft = 400,
    n_mels = 80,
    sr = target_sr,
    hop_size = 160,
    win_size = 400,
    fmin = 0,
    fmax = target_sr / 2,
    center = FALSE
  )

  # mel is (batch, mels, time), we need (batch, time, mels)
  mel <- mel$transpose(2, 3)

  # Subtract mean (per-feature normalization)
  mel <- mel - mel$mean(dim = 2, keepdim = TRUE)

  # Move to device and run inference
  mel <- mel$to(device = device)

  torch::with_no_grad({
    model(mel)
  })
}

#' Load CAMPPlus weights from safetensors
#'
#' @param model CAMPPlus model
#' @param state_dict Named list of tensors
#' @param prefix Prefix for weight keys (default "speaker_encoder.")
#' @return Model with loaded weights
#' @export
load_campplus_weights <- function(model, state_dict, prefix = "speaker_encoder.") {
  # Helper to copy weight if exists
  copy_if_exists <- function(r_param, key) {
    full_key <- paste0(prefix, key)
    if (full_key %in% names(state_dict)) {
      tryCatch({
        r_param$copy_(state_dict[[full_key]])
        return(TRUE)
      }, error = function(e) {
        warning("Failed to copy ", key, ": ", e$message)
        return(FALSE)
      })
    }
    # Try without prefix
    if (key %in% names(state_dict)) {
      tryCatch({
        r_param$copy_(state_dict[[key]])
        return(TRUE)
      }, error = function(e) FALSE)
    }
    FALSE
  }

  # ========== FCM Head ==========
  # head.conv1
  copy_if_exists(model$head$conv1$weight, "head.conv1.weight")
  # head.bn1
  copy_if_exists(model$head$bn1$weight, "head.bn1.weight")
  copy_if_exists(model$head$bn1$bias, "head.bn1.bias")
  copy_if_exists(model$head$bn1$running_mean, "head.bn1.running_mean")
  copy_if_exists(model$head$bn1$running_var, "head.bn1.running_var")

  # head.layer1 (2 BasicResBlocks)
  for (i in 0:1) {
    block_key <- sprintf("head.layer1.%d.", i)
    block <- model$head$layer1[[i + 1]]

    copy_if_exists(block$conv1$weight, paste0(block_key, "conv1.weight"))
    copy_if_exists(block$bn1$weight, paste0(block_key, "bn1.weight"))
    copy_if_exists(block$bn1$bias, paste0(block_key, "bn1.bias"))
    copy_if_exists(block$bn1$running_mean, paste0(block_key, "bn1.running_mean"))
    copy_if_exists(block$bn1$running_var, paste0(block_key, "bn1.running_var"))

    copy_if_exists(block$conv2$weight, paste0(block_key, "conv2.weight"))
    copy_if_exists(block$bn2$weight, paste0(block_key, "bn2.weight"))
    copy_if_exists(block$bn2$bias, paste0(block_key, "bn2.bias"))
    copy_if_exists(block$bn2$running_mean, paste0(block_key, "bn2.running_mean"))
    copy_if_exists(block$bn2$running_var, paste0(block_key, "bn2.running_var"))

    # Shortcut (only first block has shortcut in layer1)
    if (!is.null(block$shortcut)) {
      copy_if_exists(block$shortcut[[1]]$weight, paste0(block_key, "shortcut.0.weight"))
      copy_if_exists(block$shortcut[[2]]$weight, paste0(block_key, "shortcut.1.weight"))
      copy_if_exists(block$shortcut[[2]]$bias, paste0(block_key, "shortcut.1.bias"))
      copy_if_exists(block$shortcut[[2]]$running_mean, paste0(block_key, "shortcut.1.running_mean"))
      copy_if_exists(block$shortcut[[2]]$running_var, paste0(block_key, "shortcut.1.running_var"))
    }
  }

  # head.layer2 (2 BasicResBlocks)
  for (i in 0:1) {
    block_key <- sprintf("head.layer2.%d.", i)
    block <- model$head$layer2[[i + 1]]

    copy_if_exists(block$conv1$weight, paste0(block_key, "conv1.weight"))
    copy_if_exists(block$bn1$weight, paste0(block_key, "bn1.weight"))
    copy_if_exists(block$bn1$bias, paste0(block_key, "bn1.bias"))
    copy_if_exists(block$bn1$running_mean, paste0(block_key, "bn1.running_mean"))
    copy_if_exists(block$bn1$running_var, paste0(block_key, "bn1.running_var"))

    copy_if_exists(block$conv2$weight, paste0(block_key, "conv2.weight"))
    copy_if_exists(block$bn2$weight, paste0(block_key, "bn2.weight"))
    copy_if_exists(block$bn2$bias, paste0(block_key, "bn2.bias"))
    copy_if_exists(block$bn2$running_mean, paste0(block_key, "bn2.running_mean"))
    copy_if_exists(block$bn2$running_var, paste0(block_key, "bn2.running_var"))

    if (!is.null(block$shortcut)) {
      copy_if_exists(block$shortcut[[1]]$weight, paste0(block_key, "shortcut.0.weight"))
      copy_if_exists(block$shortcut[[2]]$weight, paste0(block_key, "shortcut.1.weight"))
      copy_if_exists(block$shortcut[[2]]$bias, paste0(block_key, "shortcut.1.bias"))
      copy_if_exists(block$shortcut[[2]]$running_mean, paste0(block_key, "shortcut.1.running_mean"))
      copy_if_exists(block$shortcut[[2]]$running_var, paste0(block_key, "shortcut.1.running_var"))
    }
  }

  # head.conv2
  copy_if_exists(model$head$conv2$weight, "head.conv2.weight")
  copy_if_exists(model$head$bn2$weight, "head.bn2.weight")
  copy_if_exists(model$head$bn2$bias, "head.bn2.bias")
  copy_if_exists(model$head$bn2$running_mean, "head.bn2.running_mean")
  copy_if_exists(model$head$bn2$running_var, "head.bn2.running_var")

  # ========== TDNN layer ==========
  copy_if_exists(model$tdnn$conv$weight, "xvector.tdnn.linear.weight")
  copy_if_exists(model$tdnn$bn$weight, "xvector.tdnn.nonlinear.batchnorm.weight")
  copy_if_exists(model$tdnn$bn$bias, "xvector.tdnn.nonlinear.batchnorm.bias")
  copy_if_exists(model$tdnn$bn$running_mean, "xvector.tdnn.nonlinear.batchnorm.running_mean")
  copy_if_exists(model$tdnn$bn$running_var, "xvector.tdnn.nonlinear.batchnorm.running_var")

  # ========== Dense Blocks ==========
  # Block 1: 12 layers
  for (i in 1:12) {
    layer_key <- sprintf("xvector.block1.tdnnd%d.", i)
    layer <- model$block1$layers[[i]]

    # nonlinear1 (batchnorm)
    copy_if_exists(layer$bn1$weight, paste0(layer_key, "nonlinear1.batchnorm.weight"))
    copy_if_exists(layer$bn1$bias, paste0(layer_key, "nonlinear1.batchnorm.bias"))
    copy_if_exists(layer$bn1$running_mean, paste0(layer_key, "nonlinear1.batchnorm.running_mean"))
    copy_if_exists(layer$bn1$running_var, paste0(layer_key, "nonlinear1.batchnorm.running_var"))

    # linear1
    copy_if_exists(layer$linear1$weight, paste0(layer_key, "linear1.weight"))

    # nonlinear2 (batchnorm)
    copy_if_exists(layer$bn2$weight, paste0(layer_key, "nonlinear2.batchnorm.weight"))
    copy_if_exists(layer$bn2$bias, paste0(layer_key, "nonlinear2.batchnorm.bias"))
    copy_if_exists(layer$bn2$running_mean, paste0(layer_key, "nonlinear2.batchnorm.running_mean"))
    copy_if_exists(layer$bn2$running_var, paste0(layer_key, "nonlinear2.batchnorm.running_var"))

    # CAM layer
    copy_if_exists(layer$cam$linear_local$weight, paste0(layer_key, "cam_layer.linear_local.weight"))
    copy_if_exists(layer$cam$linear1$weight, paste0(layer_key, "cam_layer.linear1.weight"))
    copy_if_exists(layer$cam$linear1$bias, paste0(layer_key, "cam_layer.linear1.bias"))
    copy_if_exists(layer$cam$linear2$weight, paste0(layer_key, "cam_layer.linear2.weight"))
    copy_if_exists(layer$cam$linear2$bias, paste0(layer_key, "cam_layer.linear2.bias"))
  }

  # Transit 1
  copy_if_exists(model$transit1$bn$weight, "xvector.transit1.nonlinear.batchnorm.weight")
  copy_if_exists(model$transit1$bn$bias, "xvector.transit1.nonlinear.batchnorm.bias")
  copy_if_exists(model$transit1$bn$running_mean, "xvector.transit1.nonlinear.batchnorm.running_mean")
  copy_if_exists(model$transit1$bn$running_var, "xvector.transit1.nonlinear.batchnorm.running_var")
  copy_if_exists(model$transit1$conv$weight, "xvector.transit1.linear.weight")

  # Block 2: 24 layers
  for (i in 1:24) {
    layer_key <- sprintf("xvector.block2.tdnnd%d.", i)
    layer <- model$block2$layers[[i]]

    copy_if_exists(layer$bn1$weight, paste0(layer_key, "nonlinear1.batchnorm.weight"))
    copy_if_exists(layer$bn1$bias, paste0(layer_key, "nonlinear1.batchnorm.bias"))
    copy_if_exists(layer$bn1$running_mean, paste0(layer_key, "nonlinear1.batchnorm.running_mean"))
    copy_if_exists(layer$bn1$running_var, paste0(layer_key, "nonlinear1.batchnorm.running_var"))

    copy_if_exists(layer$linear1$weight, paste0(layer_key, "linear1.weight"))

    copy_if_exists(layer$bn2$weight, paste0(layer_key, "nonlinear2.batchnorm.weight"))
    copy_if_exists(layer$bn2$bias, paste0(layer_key, "nonlinear2.batchnorm.bias"))
    copy_if_exists(layer$bn2$running_mean, paste0(layer_key, "nonlinear2.batchnorm.running_mean"))
    copy_if_exists(layer$bn2$running_var, paste0(layer_key, "nonlinear2.batchnorm.running_var"))

    copy_if_exists(layer$cam$linear_local$weight, paste0(layer_key, "cam_layer.linear_local.weight"))
    copy_if_exists(layer$cam$linear1$weight, paste0(layer_key, "cam_layer.linear1.weight"))
    copy_if_exists(layer$cam$linear1$bias, paste0(layer_key, "cam_layer.linear1.bias"))
    copy_if_exists(layer$cam$linear2$weight, paste0(layer_key, "cam_layer.linear2.weight"))
    copy_if_exists(layer$cam$linear2$bias, paste0(layer_key, "cam_layer.linear2.bias"))
  }

  # Transit 2
  copy_if_exists(model$transit2$bn$weight, "xvector.transit2.nonlinear.batchnorm.weight")
  copy_if_exists(model$transit2$bn$bias, "xvector.transit2.nonlinear.batchnorm.bias")
  copy_if_exists(model$transit2$bn$running_mean, "xvector.transit2.nonlinear.batchnorm.running_mean")
  copy_if_exists(model$transit2$bn$running_var, "xvector.transit2.nonlinear.batchnorm.running_var")
  copy_if_exists(model$transit2$conv$weight, "xvector.transit2.linear.weight")

  # Block 3: 16 layers
  for (i in 1:16) {
    layer_key <- sprintf("xvector.block3.tdnnd%d.", i)
    layer <- model$block3$layers[[i]]

    copy_if_exists(layer$bn1$weight, paste0(layer_key, "nonlinear1.batchnorm.weight"))
    copy_if_exists(layer$bn1$bias, paste0(layer_key, "nonlinear1.batchnorm.bias"))
    copy_if_exists(layer$bn1$running_mean, paste0(layer_key, "nonlinear1.batchnorm.running_mean"))
    copy_if_exists(layer$bn1$running_var, paste0(layer_key, "nonlinear1.batchnorm.running_var"))

    copy_if_exists(layer$linear1$weight, paste0(layer_key, "linear1.weight"))

    copy_if_exists(layer$bn2$weight, paste0(layer_key, "nonlinear2.batchnorm.weight"))
    copy_if_exists(layer$bn2$bias, paste0(layer_key, "nonlinear2.batchnorm.bias"))
    copy_if_exists(layer$bn2$running_mean, paste0(layer_key, "nonlinear2.batchnorm.running_mean"))
    copy_if_exists(layer$bn2$running_var, paste0(layer_key, "nonlinear2.batchnorm.running_var"))

    copy_if_exists(layer$cam$linear_local$weight, paste0(layer_key, "cam_layer.linear_local.weight"))
    copy_if_exists(layer$cam$linear1$weight, paste0(layer_key, "cam_layer.linear1.weight"))
    copy_if_exists(layer$cam$linear1$bias, paste0(layer_key, "cam_layer.linear1.bias"))
    copy_if_exists(layer$cam$linear2$weight, paste0(layer_key, "cam_layer.linear2.weight"))
    copy_if_exists(layer$cam$linear2$bias, paste0(layer_key, "cam_layer.linear2.bias"))
  }

  # Transit 3
  copy_if_exists(model$transit3$bn$weight, "xvector.transit3.nonlinear.batchnorm.weight")
  copy_if_exists(model$transit3$bn$bias, "xvector.transit3.nonlinear.batchnorm.bias")
  copy_if_exists(model$transit3$bn$running_mean, "xvector.transit3.nonlinear.batchnorm.running_mean")
  copy_if_exists(model$transit3$bn$running_var, "xvector.transit3.nonlinear.batchnorm.running_var")
  copy_if_exists(model$transit3$conv$weight, "xvector.transit3.linear.weight")

  # ========== Output ==========
  # out_nonlinear (batchnorm)
  copy_if_exists(model$out_bn$weight, "xvector.out_nonlinear.batchnorm.weight")
  copy_if_exists(model$out_bn$bias, "xvector.out_nonlinear.batchnorm.bias")
  copy_if_exists(model$out_bn$running_mean, "xvector.out_nonlinear.batchnorm.running_mean")
  copy_if_exists(model$out_bn$running_var, "xvector.out_nonlinear.batchnorm.running_var")

  # Dense layer
  copy_if_exists(model$dense$conv$weight, "xvector.dense.linear.weight")
  # Note: dense has batchnorm_ (no affine), so no weight/bias to load

  model
}
