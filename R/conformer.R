# Conformer Encoder for S3Gen
# Full implementation matching Python CosyVoice/ESPnet conformer

# ============================================================================
# Positional Encoding
# ============================================================================

#' Sinusoidal positional encoding (Espnet RelPositionalEncoding)
#'
#' Creates sinusoidal positional embeddings for use with relative position
#' attention. Includes scaling by sqrt(d_model) and adds positional embedding
#' to the input.
#'
#' @param d_model Model dimension
#' @param max_len Maximum sequence length
#' @return nn_module
espnet_rel_positional_encoding <- Rtorch::nn_module(
    "EspnetRelPositionalEncoding",

    initialize = function (d_model = 512, dropout_rate = 0.1, max_len = 5000)
    {
        self$d_model <- d_model
        self$dropout <- Rtorch::nn_dropout(dropout_rate)
        self$max_len <- max_len
        self$xscale <- sqrt(d_model) # Scale factor for input

        # Pre-compute positional encodings
        self$pe <- Rtorch::nn_buffer(
            self$create_pe(max_len, d_model)
        )
    },

    create_pe = function (max_len, d_model)
    {
        # Create positional encoding matching Python's EspnetRelPositionalEncoding
        # PE buffer has positions [+(max_len-1), ..., 0, -1, ..., -(max_len-1)]
        # Position 0 is at center (index max_len-1)

        # Position values: 0, 1, 2, ..., max_len-1
        position <- Rtorch::torch_arange(0, max_len - 1, dtype = Rtorch::torch_float32)$unsqueeze(2)

        div_term <- Rtorch::torch_exp(
            Rtorch::torch_arange(0, d_model - 1, 2, dtype = Rtorch::torch_float32) *
            (- (log(10000.0) / d_model))
        )

        # Create positive positions (sin/cos of 0, 1, 2, ..., max_len-1)
        pe_positive <- Rtorch::torch_zeros(c(max_len, d_model))
        pe_positive[, seq(1, d_model, by = 2)] <- Rtorch::torch_sin(position * div_term)
        pe_positive[, seq(2, d_model, by = 2)] <- Rtorch::torch_cos(position * div_term)

        # Create negative positions (sin/cos of 0, -1, -2, ..., -(max_len-1))
        pe_negative <- Rtorch::torch_zeros(c(max_len, d_model))
        pe_negative[, seq(1, d_model, by = 2)] <- Rtorch::torch_sin(- 1 * position * div_term)
        pe_negative[, seq(2, d_model, by = 2)] <- Rtorch::torch_cos(- 1 * position * div_term)

        # Flip positive to get [max_len-1, max_len-2, ..., 0]
        pe_positive <- Rtorch::torch_flip(pe_positive, 1L)

        # Skip position 0 from negative (it's already in positive)
        pe_negative <- pe_negative[2:max_len,]

        # Concatenate: [pe_positive_flipped, pe_negative]
        # Result: positions [+(max_len-1), ..., 0, -1, ..., -(max_len-1)]
        pe <- Rtorch::torch_cat(list(pe_positive, pe_negative), dim = 1L)

        # Add batch dimension: (1, 2*max_len-1, d_model)
        pe$unsqueeze(1)
    },

    forward = function (x, offset = 0L)
    {
        # x: (batch, time, d_model)
        # Returns: (x_scaled, pos_emb) where pos_emb is (1, 2*time-1, d_model)
        # Note: Relative positional encoding does NOT add pos_emb to x
        # The pos_emb is used in the attention layer for relative position bias
        seq_len <- x$size(2)

        # Get position embeddings centered around the sequence
        # For relative position, we need positions from -(seq_len-1) to (seq_len-1)
        # Position 0 is at R index max_len (1-indexed = Python index max_len-1)
        center <- self$max_len
        start_idx <- center - seq_len + 1L
        end_idx <- center + seq_len - 1L

        pos_emb <- self$pe[, start_idx:end_idx,]

        # Scale input by sqrt(d_model) only (no positional encoding added)
        x <- x$mul(self$xscale)

        x <- self$dropout$forward(x)

        list(x, pos_emb)
    }
)

# ============================================================================
# Linear No Subsampling (Input Embedding)
# ============================================================================

#' Linear No Subsampling layer
#'
#' Projects input to model dimension with layer norm and positional encoding.
#'
#' @param input_dim Input dimension
#' @param output_dim Output dimension
#' @param dropout_rate Dropout rate
#' @return nn_module
linear_no_subsampling <- Rtorch::nn_module(
    "LinearNoSubsampling",

    initialize = function (input_dim = 512, output_dim = 512,
                           dropout_rate = 0.1)
    {
        self$out <- Rtorch::nn_sequential(
            Rtorch::nn_linear(input_dim, output_dim),
            Rtorch::nn_layer_norm(output_dim),
            Rtorch::nn_dropout(dropout_rate)
        )
        self$pos_enc <- espnet_rel_positional_encoding(output_dim, dropout_rate)
    },

    forward = function (x, mask)
    {
        # x: (batch, time, input_dim)
        # mask: (batch, 1, time) - True for valid positions
        x <- self$out$forward(x)
        result <- self$pos_enc$forward(x)
        x <- result[[1]]
        pos_emb <- result[[2]]

        list(x, pos_emb, mask)
    }
)

# ============================================================================
# Pre-Lookahead Layer
# ============================================================================

#' Pre-Lookahead Layer
#'
#' Two causal convolutions with residual connection for look-ahead.
#'
#' @param channels Number of channels
#' @param pre_lookahead_len Look-ahead length (kernel size - 1 for conv1)
#' @return nn_module
pre_lookahead_layer <- Rtorch::nn_module(
    "PreLookaheadLayer",

    initialize = function (channels = 512, pre_lookahead_len = 3)
    {
        self$pre_lookahead_len <- pre_lookahead_len
        # conv1 kernel is lookahead + 1
        self$conv1 <- Rtorch::nn_conv1d(channels, channels, kernel_size = pre_lookahead_len + 1)
        # conv2 kernel is 3 (with 2 padding on left)
        self$conv2 <- Rtorch::nn_conv1d(channels, channels, kernel_size = 3)
    },

    forward = function (x)
    {
        # x: (batch, time, channels)
        residual <- x

        # Transpose for conv: (batch, time, channels) -> (batch, channels, time)
        h <- x$transpose(2L, 3L)$contiguous()

        # Look-ahead padding (right side)
        h <- Rtorch::nnf_pad(h, c(0L, self$pre_lookahead_len), mode = "constant", value = 0.0)

        # Conv1 with leaky relu
        h <- Rtorch::nnf_leaky_relu(self$conv1$forward(h))

        # Causal padding for conv2 (left side)
        h <- Rtorch::nnf_pad(h, c(2L, 0L), mode = "constant", value = 0.0)

        # Conv2
        h <- self$conv2$forward(h)

        # Transpose back: (batch, channels, time) -> (batch, time, channels)
        h <- h$transpose(2L, 3L)$contiguous()

        # Residual connection
        h + residual
    }
)

# ============================================================================
# Relative Position Multi-Headed Attention
# ============================================================================

#' Relative Position Multi-Headed Attention
#'
#' Multi-head attention with relative positional encodings.
#'
#' @param n_head Number of attention heads
#' @param n_feat Feature dimension
#' @param dropout_rate Dropout rate
#' @return nn_module
rel_position_attention <- Rtorch::nn_module(
    "RelPositionMultiHeadedAttention",

    initialize = function (n_head = 8, n_feat = 512, dropout_rate = 0.1)
    {
        stopifnot(n_feat %% n_head == 0)

        self$d_k <- n_feat %/% n_head
        self$h <- n_head

        # Linear projections
        self$linear_q <- Rtorch::nn_linear(n_feat, n_feat)
        self$linear_k <- Rtorch::nn_linear(n_feat, n_feat)
        self$linear_v <- Rtorch::nn_linear(n_feat, n_feat)
        self$linear_out <- Rtorch::nn_linear(n_feat, n_feat)

        # Linear for positional encoding
        self$linear_pos <- Rtorch::nn_linear(n_feat, n_feat, bias = FALSE)

        # Learnable bias for relative position
        self$pos_bias_u <- Rtorch::nn_parameter(Rtorch::torch_zeros(c(n_head, self$d_k)))
        self$pos_bias_v <- Rtorch::nn_parameter(Rtorch::torch_zeros(c(n_head, self$d_k)))

        self$dropout <- Rtorch::nn_dropout(dropout_rate)
    },

    forward = function (query, key, value, mask, pos_emb)
    {
        # query/key/value: (batch, time, n_feat)
        # mask: (batch, 1, time) or (batch, time, time)
        # pos_emb: (1, 2*time-1, n_feat)

        batch_size <- query$size(1)
        seq_len <- query$size(2)

        # Linear projections and reshape to (batch, head, time, d_k)
        q <- self$linear_q$forward(query)$view(c(batch_size, - 1, self$h, self$d_k))$transpose(2L, 3L)
        k <- self$linear_k$forward(key)$view(c(batch_size, - 1, self$h, self$d_k))$transpose(2L, 3L)
        v <- self$linear_v$forward(value)$view(c(batch_size, - 1, self$h, self$d_k))$transpose(2L, 3L)

        # Project positional encoding
        p <- self$linear_pos$forward(pos_emb)$view(c(1, - 1, self$h, self$d_k))$transpose(2L, 3L)

        # Add positional bias to query
        # q_with_bias_u: (batch, head, time, d_k)
        # pos_bias_u is (n_head, d_k), need (1, n_head, 1, d_k) for broadcasting
        q_with_bias_u <- q + self$pos_bias_u$unsqueeze(1)$unsqueeze(3)
        q_with_bias_v <- q + self$pos_bias_v$unsqueeze(1)$unsqueeze(3)

        # Content-based attention: (batch, head, time, time)
        matrix_ac <- Rtorch::torch_matmul(q_with_bias_u, k$transpose(- 2L, - 1L))

        # Position-based attention: need relative position shift
        matrix_bd <- Rtorch::torch_matmul(q_with_bias_v, p$transpose(- 2L, - 1L))
        matrix_bd <- self$rel_shift(matrix_bd)

        # Combine and scale
        scores <- (matrix_ac + matrix_bd) / sqrt(self$d_k)

        # Apply mask
        if (!is.null(mask)) {
            # Expand mask for heads: (batch, 1, 1, time) or (batch, 1, time, time)
            if (mask$dim() == 3) {
                mask <- mask$unsqueeze(2)
            }
            # Use -65504 for float16 compatibility (instead of -1e9)
            scores <- scores$masked_fill(!mask, -65504.0)
        }

        # Softmax and dropout
        attn <- Rtorch::nnf_softmax(scores, dim = - 1)
        attn <- self$dropout$forward(attn)

        # Apply attention to values
        output <- Rtorch::torch_matmul(attn, v)

        # Reshape and project output
        output <- output$transpose(2L, 3L)$contiguous()$view(c(batch_size, - 1, self$h * self$d_k))
        self$linear_out$forward(output)
    },

    rel_shift = function (x)
    {
        # x: (batch, head, time, 2*time-1)
        # Shift to align relative positions for ESPnet-style relative position attention
        batch_size <- x$size(1)
        n_head <- x$size(2)
        time <- x$size(3)
        pos_len <- x$size(4) # 2*time-1

        # Step 1: Pad left side of last dimension
        # (batch, head, time, pos_len) -> (batch, head, time, pos_len+1)
        x <- Rtorch::nnf_pad(x, c(1L, 0L))

        # Step 2: Reshape to (batch, head, pos_len+1, time)
        x <- x$view(c(batch_size, n_head, pos_len + 1L, time))

        # Step 3: Remove first row -> (batch, head, pos_len, time)
        x <- x[,, 2:(pos_len + 1L),]

        # Step 4: Reshape back to (batch, head, time, pos_len)
        x <- x$reshape(c(batch_size, n_head, time, pos_len))

        # Step 5: Take only first time columns -> (batch, head, time, time)
        x <- x[,,, 1:time]$contiguous()

        x
    }
)

# ============================================================================
# Positionwise Feed Forward
# ============================================================================

#' Positionwise Feed Forward
#'
#' Two-layer feed-forward network with SiLU activation.
#'
#' @param n_feat Input/output dimension
#' @param n_ffn Hidden dimension
#' @param dropout_rate Dropout rate
#' @return nn_module
positionwise_feedforward <- Rtorch::nn_module(
    "PositionwiseFeedForward",

    initialize = function (n_feat = 512, n_ffn = 2048, dropout_rate = 0.1)
    {
        self$w_1 <- Rtorch::nn_linear(n_feat, n_ffn)
        self$w_2 <- Rtorch::nn_linear(n_ffn, n_feat)
        self$dropout <- Rtorch::nn_dropout(dropout_rate)
    },

    forward = function (x)
    {
        # x: (batch, time, n_feat)
        x <- self$w_1$forward(x)
        x <- Rtorch::nnf_silu(x)
        x <- self$dropout$forward(x)
        x <- self$w_2$forward(x)
        x
    }
)

# ============================================================================
# Conformer Encoder Layer
# ============================================================================

#' Conformer Encoder Layer
#'
#' Single conformer block with attention and feed-forward (no convolution).
#'
#' @param n_feat Feature dimension
#' @param n_head Number of attention heads
#' @param n_ffn Feed-forward hidden dimension
#' @param dropout_rate Dropout rate
#' @return nn_module
conformer_encoder_layer <- Rtorch::nn_module(
    "ConformerEncoderLayer",

    initialize = function (n_feat = 512, n_head = 8, n_ffn = 2048,
                           dropout_rate = 0.1)
    {
        # Self-attention
        self$self_attn <- rel_position_attention(n_head, n_feat, dropout_rate)

        # Feed-forward
        self$feed_forward <- positionwise_feedforward(n_feat, n_ffn, dropout_rate)

        # Layer norms (eps=1e-12 to match Python)
        self$norm_mha <- Rtorch::nn_layer_norm(n_feat, eps = 1e-12)
        self$norm_ff <- Rtorch::nn_layer_norm(n_feat, eps = 1e-12)

        # Dropout
        self$dropout <- Rtorch::nn_dropout(dropout_rate)
    },

    forward = function (x, mask, pos_emb, mask_pad = NULL)
    {
        # x: (batch, time, n_feat)
        # mask: (batch, 1, time) attention mask
        # pos_emb: (1, 2*time-1, n_feat) positional encoding
        # mask_pad: (batch, 1, time) padding mask

        # Multi-head attention with pre-norm
        residual <- x
        x <- self$norm_mha$forward(x)
        x <- self$self_attn$forward(x, x, x, mask, pos_emb)
        x <- self$dropout$forward(x)
        x <- residual + x

        # Feed-forward with pre-norm
        residual <- x
        x <- self$norm_ff$forward(x)
        x <- self$feed_forward$forward(x)
        x <- self$dropout$forward(x)
        x <- residual + x

        # Apply padding mask
        if (!is.null(mask_pad) && mask_pad$dim() > 0) {
            x <- x$masked_fill(!mask_pad$transpose(2L, 3L), 0.0)
        }

        list(x, mask, NULL, NULL)
    }
)

# ============================================================================
# Upsample 1D
# ============================================================================

#' Upsample 1D
#'
#' 2x upsampling using interpolation + convolution.
#'
#' @param channels Number of channels
#' @param stride Upsample factor
#' @return nn_module
upsample_1d <- Rtorch::nn_module(
    "Upsample1D",

    initialize = function (channels = 512, stride = 2L)
    {
        self$stride <- stride
        self$conv <- Rtorch::nn_conv1d(channels, channels, kernel_size = 5)
    },

    forward = function (x, x_lens)
    {
        # x: (batch, channels, time)
        # x_lens: (batch,)

        # Interpolate 2x
        x <- Rtorch::nnf_interpolate(x, scale_factor = as.double(self$stride), mode = "nearest")

        # Pad left (stride * 2)
        x <- Rtorch::nnf_pad(x, c(self$stride * 2L, 0L), value = 0.0)

        # Convolution
        x <- self$conv$forward(x)

        # Update lengths
        new_lens <- x_lens * self$stride

        list(x, new_lens)
    }
)

# ============================================================================
# Upsample Conformer Encoder (Full)
# ============================================================================

#' Upsample Conformer Encoder
#'
#' Full conformer encoder matching Python UpsampleConformerEncoder.
#'
#' @param input_size Input dimension
#' @param output_size Output dimension
#' @param num_blocks Number of conformer blocks before upsample
#' @param num_up_blocks Number of conformer blocks after upsample
#' @param n_head Number of attention heads
#' @param n_ffn Feed-forward hidden dimension
#' @param dropout_rate Dropout rate
#' @param pre_lookahead_len Look-ahead length
#' @return nn_module
upsample_conformer_encoder_full <- Rtorch::nn_module(
    "UpsampleConformerEncoder",

    initialize = function (input_size = 512, output_size = 512, num_blocks = 6,
                           num_up_blocks = 4, n_head = 8, n_ffn = 2048,
                           dropout_rate = 0.1, pre_lookahead_len = 3)
    {
        self$input_size <- input_size
        self$output_size_val <- output_size
        self$pre_lookahead_len <- pre_lookahead_len

        # Input embedding with positional encoding
        self$embed <- linear_no_subsampling(input_size, output_size, dropout_rate)

        # Pre-lookahead layer
        self$pre_lookahead_layer <- pre_lookahead_layer(output_size, pre_lookahead_len)

        # First set of conformer blocks (before upsample)
        self$encoders <- Rtorch::nn_module_list(
            lapply(seq_len(num_blocks), function (i)
            {
                    conformer_encoder_layer(output_size, n_head, n_ffn, dropout_rate)
                })
        )

        # Upsample layer
        self$up_layer <- upsample_1d(output_size, stride = 2L)

        # Second embedding for after upsample
        self$up_embed <- linear_no_subsampling(output_size, output_size, dropout_rate)

        # Second set of conformer blocks (after upsample)
        self$up_encoders <- Rtorch::nn_module_list(
            lapply(seq_len(num_up_blocks), function (i)
            {
                    conformer_encoder_layer(output_size, n_head, n_ffn, dropout_rate)
                })
        )

        # Final layer norm
        self$after_norm <- Rtorch::nn_layer_norm(output_size)

        # Chunk parameters (not used in inference, but stored)
        self$use_dynamic_chunk <- FALSE
        self$use_dynamic_left_chunk <- FALSE
        self$static_chunk_size <- 0L
    },

    output_size = function ()
    {
        self$output_size_val
    },

    forward = function (x, x_lens)
    {
        # x: (batch, time, features)
        # x_lens: (batch,)

        device <- x$device
        T <- x$size(2)

        # Create attention mask
        masks <- self$make_pad_mask(x_lens, T, device)$unsqueeze(2)
        masks <- !masks# True for valid positions

        # Input embedding with positional encoding
        embed_result <- self$embed$forward(x, masks)
        xs <- embed_result[[1]]
        pos_emb <- embed_result[[2]]
        masks <- embed_result[[3]]

        # Pre-lookahead layer
        xs <- self$pre_lookahead_layer$forward(xs)

        # First set of conformer blocks
        mask_pad <- masks
        for (i in seq_along(self$encoders)) {
            result <- self$encoders[[i]]$forward(xs, masks, pos_emb, mask_pad)
            xs <- result[[1]]
        }

        # Transpose for upsample: (batch, time, feat) -> (batch, feat, time)
        xs <- xs$transpose(2L, 3L)$contiguous()

        # Upsample
        up_result <- self$up_layer$forward(xs, x_lens)
        xs <- up_result[[1]]
        xs_lens <- up_result[[2]]

        # Transpose back: (batch, feat, time) -> (batch, time, feat)
        xs <- xs$transpose(2L, 3L)$contiguous()

        # Create new mask for upsampled sequence
        T_up <- xs$size(2)
        masks_up <- self$make_pad_mask(xs_lens, T_up, device)$unsqueeze(2)
        masks_up <- !masks_up

        # Second embedding with positional encoding
        up_embed_result <- self$up_embed$forward(xs, masks_up)
        xs <- up_embed_result[[1]]
        pos_emb_up <- up_embed_result[[2]]
        masks_up <- up_embed_result[[3]]

        # Second set of conformer blocks
        mask_pad_up <- masks_up
        for (i in seq_along(self$up_encoders)) {
            result <- self$up_encoders[[i]]$forward(xs, masks_up, pos_emb_up, mask_pad_up)
            xs <- result[[1]]
        }

        # Final layer norm
        xs <- self$after_norm$forward(xs)

        list(xs, masks_up)
    },

    make_pad_mask = function (lengths, max_len, device)
    {
        # Create boolean mask where TRUE = padding position
        batch_size <- lengths$size(1)

        # R torch_arange(0, n-1) creates 0..n-1 (n values)
        range_tensor <- Rtorch::torch_arange(0, max_len - 1, device = device, dtype = Rtorch::torch_long)
        range_tensor <- range_tensor$unsqueeze(2) # (max_len, 1) - unsqueeze(2) adds dim at position 2

        lengths_expand <- lengths$view(c(1, batch_size)) # (1, batch)
        mask <- range_tensor >= lengths_expand# (max_len, batch)
        mask$transpose(1L, 2L) # (batch, max_len)
    }
)

# ============================================================================
# Weight Loading
# ============================================================================

#' Load Conformer Encoder weights
#'
#' @param model Conformer encoder module
#' @param state_dict State dictionary
#' @param prefix Key prefix (e.g., "flow.encoder.")
load_conformer_encoder_weights <- function (model, state_dict,
                                            prefix = "flow.encoder.")
{
    copy_if_exists <- function (r_param, key)
    {
        full_key <- paste0(prefix, key)
        if (full_key %in% names(state_dict)) {
            tryCatch({
                    # Must use with_no_grad to avoid autograd error on leaf variables
                    Rtorch::with_no_grad({
                            r_param$copy_(state_dict[[full_key]])
                        })
                    TRUE
                }, error = function (e)
                {
                    warning("Failed to copy ", full_key, ": ", e$message)
                    FALSE
                })
        } else {
            FALSE
        }
    }

    # Input embedding (embed.out)
    copy_if_exists(model$embed$out[[1]]$weight, "embed.out.0.weight")
    copy_if_exists(model$embed$out[[1]]$bias, "embed.out.0.bias")
    copy_if_exists(model$embed$out[[2]]$weight, "embed.out.1.weight")
    copy_if_exists(model$embed$out[[2]]$bias, "embed.out.1.bias")

    # Pre-lookahead layer
    copy_if_exists(model$pre_lookahead_layer$conv1$weight, "pre_lookahead_layer.conv1.weight")
    copy_if_exists(model$pre_lookahead_layer$conv1$bias, "pre_lookahead_layer.conv1.bias")
    copy_if_exists(model$pre_lookahead_layer$conv2$weight, "pre_lookahead_layer.conv2.weight")
    copy_if_exists(model$pre_lookahead_layer$conv2$bias, "pre_lookahead_layer.conv2.bias")

    # Conformer encoder blocks
    for (i in seq_along(model$encoders)) {
        idx <- i - 1# Python uses 0-indexing
        p <- paste0("encoders.", idx, ".")

        # Self-attention
        copy_if_exists(model$encoders[[i]]$self_attn$linear_q$weight, paste0(p, "self_attn.linear_q.weight"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_q$bias, paste0(p, "self_attn.linear_q.bias"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_k$weight, paste0(p, "self_attn.linear_k.weight"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_k$bias, paste0(p, "self_attn.linear_k.bias"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_v$weight, paste0(p, "self_attn.linear_v.weight"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_v$bias, paste0(p, "self_attn.linear_v.bias"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_out$weight, paste0(p, "self_attn.linear_out.weight"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_out$bias, paste0(p, "self_attn.linear_out.bias"))
        copy_if_exists(model$encoders[[i]]$self_attn$linear_pos$weight, paste0(p, "self_attn.linear_pos.weight"))
        copy_if_exists(model$encoders[[i]]$self_attn$pos_bias_u, paste0(p, "self_attn.pos_bias_u"))
        copy_if_exists(model$encoders[[i]]$self_attn$pos_bias_v, paste0(p, "self_attn.pos_bias_v"))

        # Feed-forward
        copy_if_exists(model$encoders[[i]]$feed_forward$w_1$weight, paste0(p, "feed_forward.w_1.weight"))
        copy_if_exists(model$encoders[[i]]$feed_forward$w_1$bias, paste0(p, "feed_forward.w_1.bias"))
        copy_if_exists(model$encoders[[i]]$feed_forward$w_2$weight, paste0(p, "feed_forward.w_2.weight"))
        copy_if_exists(model$encoders[[i]]$feed_forward$w_2$bias, paste0(p, "feed_forward.w_2.bias"))

        # Layer norms
        copy_if_exists(model$encoders[[i]]$norm_mha$weight, paste0(p, "norm_mha.weight"))
        copy_if_exists(model$encoders[[i]]$norm_mha$bias, paste0(p, "norm_mha.bias"))
        copy_if_exists(model$encoders[[i]]$norm_ff$weight, paste0(p, "norm_ff.weight"))
        copy_if_exists(model$encoders[[i]]$norm_ff$bias, paste0(p, "norm_ff.bias"))
    }

    # Upsample layer
    copy_if_exists(model$up_layer$conv$weight, "up_layer.conv.weight")
    copy_if_exists(model$up_layer$conv$bias, "up_layer.conv.bias")

    # Up embedding
    copy_if_exists(model$up_embed$out[[1]]$weight, "up_embed.out.0.weight")
    copy_if_exists(model$up_embed$out[[1]]$bias, "up_embed.out.0.bias")
    copy_if_exists(model$up_embed$out[[2]]$weight, "up_embed.out.1.weight")
    copy_if_exists(model$up_embed$out[[2]]$bias, "up_embed.out.1.bias")

    # Up encoder blocks
    for (i in seq_along(model$up_encoders)) {
        idx <- i - 1
        p <- paste0("up_encoders.", idx, ".")

        # Self-attention
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_q$weight, paste0(p, "self_attn.linear_q.weight"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_q$bias, paste0(p, "self_attn.linear_q.bias"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_k$weight, paste0(p, "self_attn.linear_k.weight"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_k$bias, paste0(p, "self_attn.linear_k.bias"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_v$weight, paste0(p, "self_attn.linear_v.weight"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_v$bias, paste0(p, "self_attn.linear_v.bias"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_out$weight, paste0(p, "self_attn.linear_out.weight"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_out$bias, paste0(p, "self_attn.linear_out.bias"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$linear_pos$weight, paste0(p, "self_attn.linear_pos.weight"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$pos_bias_u, paste0(p, "self_attn.pos_bias_u"))
        copy_if_exists(model$up_encoders[[i]]$self_attn$pos_bias_v, paste0(p, "self_attn.pos_bias_v"))

        # Feed-forward
        copy_if_exists(model$up_encoders[[i]]$feed_forward$w_1$weight, paste0(p, "feed_forward.w_1.weight"))
        copy_if_exists(model$up_encoders[[i]]$feed_forward$w_1$bias, paste0(p, "feed_forward.w_1.bias"))
        copy_if_exists(model$up_encoders[[i]]$feed_forward$w_2$weight, paste0(p, "feed_forward.w_2.weight"))
        copy_if_exists(model$up_encoders[[i]]$feed_forward$w_2$bias, paste0(p, "feed_forward.w_2.bias"))

        # Layer norms
        copy_if_exists(model$up_encoders[[i]]$norm_mha$weight, paste0(p, "norm_mha.weight"))
        copy_if_exists(model$up_encoders[[i]]$norm_mha$bias, paste0(p, "norm_mha.bias"))
        copy_if_exists(model$up_encoders[[i]]$norm_ff$weight, paste0(p, "norm_ff.weight"))
        copy_if_exists(model$up_encoders[[i]]$norm_ff$bias, paste0(p, "norm_ff.bias"))
    }

    # Final layer norm
    copy_if_exists(model$after_norm$weight, "after_norm.weight")
    copy_if_exists(model$after_norm$bias, "after_norm.bias")

    model
}

