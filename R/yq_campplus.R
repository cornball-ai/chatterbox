# anvl/yunque port of the CAMPPlus speaker encoder (xvector, 192-d).
# Torch-free; yq_ marks the anvl/XLA implementation. Composed from yunque
# conv1d/conv2d/batch_norm; avg_pool1d (seg pooling) and statistics pooling
# are reshape + nv_reduce_sum glue.

.yq_relu <- function(x) anvl::nv_max(x, 0)

# Batch-norm from a {w,b,rm,rv} weight bundle (inference mode, eps 1e-5).
.yq_bn <- function(x, b) {
  yunque::batch_norm(x, b$rm, b$rv, b$w, b$b)
}

# Segment average pooling: avg_pool1d(kernel = stride = seg_len,
# ceil_mode = TRUE), then broadcast each segment mean back over its span
# (partial final window averages only its valid frames). x is [B, C, L].
.yq_seg_pooling <- function(x, seg_len = 100L) {
  s <- anvl::shape(x)
  L <- s[3L]
  n_full <- L %/% seg_len
  rem <- L %% seg_len
  seg_mean <- function(from, to) {
    w <- to - from + 1L
    sub <- anvl::nv_static_slice(x,
      start_indices = c(1L, 1L, from),
      limit_indices = c(s[1L], s[2L], to),
      strides = c(1L, 1L, 1L))
    m <- anvl::nv_reduce_sum(sub, dims = 3L, drop = FALSE) / w
    anvl::nv_broadcast_to(m, c(s[1L], s[2L], w))
  }
  parts <- list()
  if (n_full > 0L) {
    for (i in seq_len(n_full)) {
      parts[[length(parts) + 1L]] <- seg_mean((i - 1L) * seg_len + 1L,
        i * seg_len)
    }
  }
  if (rem > 0L) {
    parts[[length(parts) + 1L]] <- seg_mean(n_full * seg_len + 1L, L)
  }
  if (length(parts) == 1L) {
    return(parts[[1L]])
  }
  do.call(anvl::nv_concatenate, c(parts, list(dimension = 3L)))
}

# Statistics pooling: mean concat unbiased-std over time. [B, C, L] -> [B, 2C].
.yq_stats_pool <- function(x) {
  s <- anvl::shape(x)
  L <- s[3L]
  mean <- anvl::nv_reduce_sum(x, dims = 3L, drop = FALSE) / L
  xc <- x - anvl::nv_broadcast_to(mean, s)
  var <- anvl::nv_reduce_sum(xc * xc, dims = 3L, drop = FALSE) / (L - 1L)
  std <- anvl::nv_sqrt(var)
  mean <- anvl::nv_reshape(mean, c(s[1L], s[2L]))
  std <- anvl::nv_reshape(std, c(s[1L], s[2L]))
  anvl::nv_concatenate(mean, std, dimension = 2L)
}

# FCM BasicResBlock: two conv2d/bn, optional 1x1 stride shortcut, relu.
.yq_res_block <- function(x, b, stride) {
  out <- .yq_relu(.yq_bn(yunque::conv2d(x, b$conv1, stride = c(stride, 1L),
    padding = 1L), b$bn1))
  out <- .yq_bn(yunque::conv2d(out, b$conv2, stride = 1L, padding = 1L), b$bn2)
  if (!is.null(b$shortcut)) {
    sc <- .yq_bn(yunque::conv2d(x, b$shortcut$conv, stride = c(stride, 1L),
      padding = 0L), b$shortcut$bn)
    out <- out + sc
  } else {
    out <- out + x
  }
  .yq_relu(out)
}

# FCM head: [B, F, T] -> [B, C*H, T].
.yq_fcm <- function(x, w) {
  x <- anvl::nv_unsqueeze(x, 2L) # [B, 1, F, T]
  out <- .yq_relu(.yq_bn(yunque::conv2d(x, w$conv1, stride = 1L,
    padding = 1L), w$bn1))
  out <- .yq_res_block(out, w$layer1[[1L]], 2L)
  out <- .yq_res_block(out, w$layer1[[2L]], 1L)
  out <- .yq_res_block(out, w$layer2[[1L]], 2L)
  out <- .yq_res_block(out, w$layer2[[2L]], 1L)
  out <- .yq_relu(.yq_bn(yunque::conv2d(out, w$conv2, stride = c(2L, 1L),
    padding = 1L), w$bn2))
  s <- anvl::shape(out)
  anvl::nv_reshape(out, c(s[1L], s[2L] * s[3L], s[4L]))
}

# CAM (context-aware masking) layer. x is [B, bn_channels, L].
.yq_cam <- function(x, cw, padding, dilation) {
  y <- yunque::conv1d(x, cw$linear_local, stride = 1L, padding = padding,
    dilation = dilation)
  s <- anvl::shape(x)
  gmean <- anvl::nv_reduce_sum(x, dims = 3L, drop = FALSE) / s[3L]
  seg <- .yq_seg_pooling(x, 100L)
  context <- anvl::nv_broadcast_to(gmean, anvl::shape(seg)) + seg
  context <- .yq_relu(yunque::conv1d(context, cw$l1w, cw$l1b))
  m <- anvl::nv_logistic(yunque::conv1d(context, cw$l2w, cw$l2b))
  y * m
}

# CAMDenseTDNNLayer.
.yq_dense_layer <- function(x, lw, dilation) {
  out <- .yq_relu(.yq_bn(x, lw$bn1))
  out <- yunque::conv1d(out, lw$linear1)
  out <- .yq_relu(.yq_bn(out, lw$bn2))
  .yq_cam(out, lw$cam, padding = dilation, dilation = dilation)
}

# CAMDenseTDNNBlock: dense (concatenative) connections.
.yq_dense_block <- function(x, layers, dilation) {
  for (lw in layers) {
    out <- .yq_dense_layer(x, lw, dilation)
    x <- anvl::nv_concatenate(x, out, dimension = 2L)
  }
  x
}

# TransitLayer: relu(bn) then 1x1 conv (no bias).
.yq_transit <- function(x, tw) {
  yunque::conv1d(.yq_relu(.yq_bn(x, tw$bn)), tw$conv)
}

#' Load CAMPPlus speaker-encoder weights (anvl)
#'
#' @param path Path to s3gen.safetensors.
#' @param prefix Key prefix (default \code{"speaker_encoder."}).
#' @return Nested list of AnvlArray weights for \code{\link{yq_campplus}}.
#' @export
yq_campplus_load_weights <- function(path, prefix = "speaker_encoder.") {
  st <- yunque::st_open(path)
  on.exit(yunque::st_close(st))
  nv <- function(k) {
    anvl::nv_array(yunque::st_read(st, paste0(prefix, k), transpose = FALSE),
      dtype = "f32")
  }
  bn <- function(k) {
    list(w = nv(paste0(k, ".weight")), b = nv(paste0(k, ".bias")),
      rm = nv(paste0(k, ".running_mean")), rv = nv(paste0(k, ".running_var")))
  }
  res_block <- function(k, has_shortcut) {
    b <- list(
      conv1 = nv(paste0(k, ".conv1.weight")), bn1 = bn(paste0(k, ".bn1")),
      conv2 = nv(paste0(k, ".conv2.weight")), bn2 = bn(paste0(k, ".bn2")),
      shortcut = NULL)
    if (has_shortcut) {
      b$shortcut <- list(conv = nv(paste0(k, ".shortcut.0.weight")),
        bn = bn(paste0(k, ".shortcut.1")))
    }
    b
  }
  dense_layer <- function(k) {
    list(
      bn1 = bn(paste0(k, ".nonlinear1.batchnorm")),
      linear1 = nv(paste0(k, ".linear1.weight")),
      bn2 = bn(paste0(k, ".nonlinear2.batchnorm")),
      cam = list(
        linear_local = nv(paste0(k, ".cam_layer.linear_local.weight")),
        l1w = nv(paste0(k, ".cam_layer.linear1.weight")),
        l1b = nv(paste0(k, ".cam_layer.linear1.bias")),
        l2w = nv(paste0(k, ".cam_layer.linear2.weight")),
        l2b = nv(paste0(k, ".cam_layer.linear2.bias"))))
  }
  block <- function(name, n) {
    lapply(seq_len(n), function(i)
      dense_layer(sprintf("xvector.%s.tdnnd%d", name, i)))
  }
  transit <- function(name) {
    list(bn = bn(sprintf("xvector.%s.nonlinear.batchnorm", name)),
      conv = nv(sprintf("xvector.%s.linear.weight", name)))
  }
  list(
    head = list(
      conv1 = nv("head.conv1.weight"), bn1 = bn("head.bn1"),
      layer1 = list(res_block("head.layer1.0", TRUE),
        res_block("head.layer1.1", FALSE)),
      layer2 = list(res_block("head.layer2.0", TRUE),
        res_block("head.layer2.1", FALSE)),
      conv2 = nv("head.conv2.weight"), bn2 = bn("head.bn2")),
    tdnn = list(conv = nv("xvector.tdnn.linear.weight"),
      bn = bn("xvector.tdnn.nonlinear.batchnorm")),
    block1 = block("block1", 12L), transit1 = transit("transit1"),
    block2 = block("block2", 24L), transit2 = transit("transit2"),
    block3 = block("block3", 16L), transit3 = transit("transit3"),
    out_bn = bn("xvector.out_nonlinear.batchnorm"),
    dense = list(conv = nv("xvector.dense.linear.weight"),
      rm = nv("xvector.dense.nonlinear.batchnorm.running_mean"),
      rv = nv("xvector.dense.nonlinear.batchnorm.running_var")))
}

#' CAMPPlus speaker encoder forward (anvl)
#'
#' Torch-free port of \code{campplus$forward}: FCM head, TDNN, three
#' CAM-dense-TDNN blocks + transit layers, statistics pooling, dense
#' projection to a 192-d xvector.
#'
#' @param mels AnvlArray \code{[B, T, 80]} (kaldi-fbank mels, batch-first).
#' @param w Weights from \code{\link{yq_campplus_load_weights}}.
#'
#' @return AnvlArray \code{[B, 192]} speaker embedding.
#'
#' @export
yq_campplus <- function(mels, w) {
  x <- anvl::nv_transpose(mels, c(1L, 3L, 2L)) # [B, F, T]
  x <- .yq_fcm(x, w$head)
  x <- .yq_relu(.yq_bn(yunque::conv1d(x, w$tdnn$conv, stride = 2L,
    padding = 2L), w$tdnn$bn))
  x <- .yq_transit(.yq_dense_block(x, w$block1, 1L), w$transit1)
  x <- .yq_transit(.yq_dense_block(x, w$block2, 2L), w$transit2)
  x <- .yq_transit(.yq_dense_block(x, w$block3, 2L), w$transit3)
  x <- .yq_relu(.yq_bn(x, w$out_bn))
  x <- .yq_stats_pool(x) # [B, 2C]
  x <- anvl::nv_unsqueeze(x, 3L) # [B, 2C, 1]
  x <- yunque::conv1d(x, w$dense$conv)
  x <- yunque::batch_norm(x, w$dense$rm, w$dense$rv) # affine = FALSE
  s <- anvl::shape(x)
  anvl::nv_reshape(x, c(s[1L], s[2L]))
}
