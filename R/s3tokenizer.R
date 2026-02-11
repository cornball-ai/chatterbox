# S3Tokenizer for chatterbox
# Speech tokenizer that converts audio to discrete speech tokens
# Based on S3TokenizerV2 from xingchensong/S3Tokenizer

# Constants
S3_SR <- 16000# Sample rate
S3_HOP <- 160# STFT hop (100 frames/sec)
S3_TOKEN_RATE <- 25# Tokens per second
SPEECH_VOCAB_SIZE <- 6561# 3^8 codebook size

# ============================================================================
# Configuration
# ============================================================================

#' S3Tokenizer model configuration
#'
#' @param n_mels Number of mel bins (default 128)
#' @param n_audio_state Hidden state dimension (default 1280)
#' @param n_audio_head Number of attention heads (default 20)
#' @param n_audio_layer Number of transformer layers (default 6)
#' @param n_codebook_size Codebook size (default 6561 = 3^8)
#' @return Configuration list
s3_tokenizer_config <- function (n_mels = 128, n_audio_state = 1280,
                                 n_audio_head = 20, n_audio_layer = 6,
                                 n_codebook_size = 6561)
{
    list(
        n_mels = n_mels,
        n_audio_ctx = 1500,
        n_audio_state = n_audio_state,
        n_audio_head = n_audio_head,
        n_audio_layer = n_audio_layer,
        n_codebook_size = n_codebook_size
    )
}

# ============================================================================
# Utility Functions
# ============================================================================

#' Create non-padding mask
#'
#' @param lengths Tensor of sequence lengths
#' @param max_len Maximum sequence length
#' @return Boolean mask tensor (TRUE for valid positions)
make_non_pad_mask_s3 <- function (lengths, max_len)
{
    batch_size <- lengths$size(1)
    device <- lengths$device

    # Create range tensor (0 to max_len-1)
    # Note: R torch_arange(0, n) is inclusive on both ends, so use 0 to n-1
    range_tensor <- Rtorch::torch_arange(0, max_len - 1, device = device, dtype = Rtorch::torch_long)

    # Broadcast comparison: range < lengths
    lengths <- lengths$view(c(- 1, 1))
    range_tensor < lengths
}

#' Convert mask to attention bias
#'
#' @param mask Boolean mask
#' @param dtype Target dtype
#' @return Attention bias tensor
mask_to_bias <- function (mask, dtype)
{
    # Convert boolean mask to attention bias (-65504 for masked positions)
    # Use -65504 instead of -Inf for float16 compatibility
    bias <- Rtorch::torch_zeros_like(mask, dtype = dtype)
    bias[!mask] <- -65504.0
    bias
}

#' Precompute rotary position embedding frequencies
#'
#' @param dim Dimension (head_dim)
#' @param end Maximum sequence length
#' @param theta Base frequency
#' @return Complex frequency tensor
precompute_freqs_cis <- function (dim, end, theta = 10000.0)
{
    # Compute inverse frequencies
    # R torch_arange is inclusive, so use end= or subtract 1
    # torch_arange(start=0, end=dim, step=2) gives 0,2,4,...,dim in R vs 0,2,4,...,dim-2 in Python
    freqs <- 1.0 / (theta ^ (Rtorch::torch_arange(start = 0, end = dim - 1, step = 2, dtype = Rtorch::torch_float32)[1:(dim %/% 2)] / dim))

    # Compute position indices (0 to end-1)
    t <- Rtorch::torch_arange(0, end - 1, dtype = Rtorch::torch_float32)

    # Outer product
    freqs <- Rtorch::torch_outer(t, freqs)

    # Create complex exponential using polar form
    freqs_cis <- Rtorch::torch_polar(Rtorch::torch_ones_like(freqs), freqs)

    # Concatenate for both halves
    Rtorch::torch_cat(list(freqs_cis, freqs_cis), dim = - 1)
}

#' Apply rotary position embeddings
#'
#' @param xq Query tensor
#' @param xk Key tensor
#' @param freqs_cis Precomputed frequencies
#' @return List with rotated q and k
apply_rotary_emb_s3 <- function (xq, xk, freqs_cis)
{
    # Get cos and sin from complex frequencies
    real_part <- Rtorch::torch_view_as_real(freqs_cis)
    cos_vals <- real_part[,, 1]
    sin_vals <- real_part[,, 2]

    cos_vals <- cos_vals$unsqueeze(1)$unsqueeze(3)
    sin_vals <- sin_vals$unsqueeze(1)$unsqueeze(3)

    D <- xq$size(4)
    half <- D %/% 2

    # Rotate query
    xq_l <- xq[,,, 1:half]
    xq_r <- xq[,,, (half + 1) :D]
    xq_rot <- Rtorch::torch_cat(list(- xq_r, xq_l), dim = - 1)
    xq_out <- xq * cos_vals + xq_rot * sin_vals

    # Rotate key
    xk_l <- xk[,,, 1:half]
    xk_r <- xk[,,, (half + 1) :D]
    xk_rot <- Rtorch::torch_cat(list(- xk_r, xk_l), dim = - 1)
    xk_out <- xk * cos_vals + xk_rot * sin_vals

    list(q = xq_out, k = xk_out)
}

# ============================================================================
# Mel Spectrogram for S3Tokenizer
# ============================================================================

#' Compute log mel spectrogram for S3Tokenizer
#'
#' @param audio Audio tensor (batch, samples)
#' @param mel_filters Pre-computed mel filterbank
#' @param window Hann window
#' @param n_fft FFT size (default 400)
#' @param device Device
#' @return Log mel spectrogram (batch, n_mels, time)
s3_log_mel_spectrogram <- function (audio, mel_filters, window, n_fft = 400,
                                    device = "cpu")
{
    if (!inherits(audio, "torch_tensor")) {
        audio <- Rtorch::torch_tensor(audio, dtype = Rtorch::torch_float32)
    }

    audio <- audio$to(device = device)
    window <- window$to(device = device)
    mel_filters <- mel_filters$to(device = device)

    # Compute STFT
    stft <- Rtorch::torch_stft(
        audio,
        n_fft = n_fft,
        hop_length = S3_HOP,
        window = window,
        return_complex = TRUE
    )

    # Magnitude squared, drop last frame
    magnitudes <- Rtorch::torch_abs(stft[,, 1:(stft$size(3) - 1)])$pow(2)

    # Apply mel filterbank
    mel_spec <- Rtorch::torch_matmul(mel_filters, magnitudes)

    # Log compression with dynamic range compression
    log_spec <- Rtorch::torch_clamp(mel_spec, min = 1e-10)$log10()
    max_spec <- log_spec$max()
    log_spec <- Rtorch::torch_maximum(log_spec, max_spec - 8.0)
    log_spec <- (log_spec + 4.0) / 4.0

    log_spec
}

# ============================================================================
# FSQ Codebook (Finite Scalar Quantization)
# ============================================================================

#' FSQ Codebook module
#'
#' @param dim Input dimension (n_audio_state)
#' @param level Quantization level (default 3)
#' @return nn_module
fsq_codebook <- Rtorch::nn_module(
    "FSQCodebook",

    initialize = function (dim, level = 3L)
    {
        self$level <- level
        self$project_down <- Rtorch::nn_linear(dim, 8L)
    },

    encode = function (x)
    {
        # x: (batch, time, dim)
        x_shape <- x$shape

        # Flatten batch and time
        x <- x$view(c(- 1, x_shape[length(x_shape)]))

        # Project down to 8 dimensions
        h <- self$project_down$forward(x)$to(dtype = Rtorch::torch_float32)

        # Quantize with tanh
        h <- h$tanh()
        h <- h * 0.9990000128746033
        h <- h$round() + 1# Range [0, 2] for level=3

        # Compute indices using powers of level
        device <- x$device
        # R torch_arange(0, 8) is inclusive (0-8); use 0 to 7 for 8 values
        powers <- Rtorch::torch_pow(
            self$level,
            Rtorch::torch_arange(0, 7, device = device, dtype = h$dtype)
        )

        # Sum weighted by powers: gives unique index for each combination
        mu <- Rtorch::torch_sum(h * powers$unsqueeze(1), dim = - 1)

        # Reshape back to (batch, time)
        mu$view(c(x_shape[1], x_shape[2]))$to(dtype = Rtorch::torch_long)
    }
)

#' FSQ Vector Quantization wrapper
#'
#' @param dim Input dimension
#' @param codebook_size Codebook size (must be 6561 = 3^8)
#' @return nn_module
fsq_vector_quantization <- Rtorch::nn_module(
    "FSQVectorQuantization",

    initialize = function (dim, codebook_size = 6561L)
    {
        stopifnot(codebook_size == 3L ^ 8L)
        self$codebook <- fsq_codebook(dim, level = 3L)
        self$codebook_size <- codebook_size
    },

    encode = function (x)
    {
        self$codebook$encode(x)
    }
)

# ============================================================================
# Multi-Head Attention with FSMN
# ============================================================================

#' Multi-Head Attention base module
#'
#' @param n_state Hidden dimension
#' @param n_head Number of heads
#' @return nn_module
s3_multi_head_attention <- Rtorch::nn_module(
    "S3MultiHeadAttention",

    initialize = function (n_state, n_head)
    {
        self$n_head <- n_head
        self$n_state <- n_state
        self$head_dim <- n_state %/% n_head

        self$query <- Rtorch::nn_linear(n_state, n_state)
        self$key <- Rtorch::nn_linear(n_state, n_state, bias = FALSE)
        self$value <- Rtorch::nn_linear(n_state, n_state)
        self$out <- Rtorch::nn_linear(n_state, n_state)
    },

    forward = function (x, mask = NULL)
    {
        q <- self$query$forward(x)
        k <- self$key$forward(x)
        v <- self$value$forward(x)

        wv <- self$qkv_attention(q, k, v, mask)
        self$out$forward(wv)
    },

    qkv_attention = function (q, k, v, mask = NULL)
    {
        batch_size <- q$size(1)
        seq_len <- q$size(2)
        D <- q$size(3)
        scale <- (D / self$n_head) ^ (- 0.25)

        # Reshape to (batch, heads, seq, head_dim)
        q <- q$view(c(batch_size, seq_len, self$n_head, - 1))$permute(c(1, 3, 2, 4)) * scale
        k <- k$view(c(batch_size, seq_len, self$n_head, - 1))$permute(c(1, 3, 4, 2)) * scale
        v <- v$view(c(batch_size, seq_len, self$n_head, - 1))$permute(c(1, 3, 2, 4))

        # Attention
        qk <- Rtorch::torch_matmul(q, k)
        if (!is.null(mask)) {
            qk <- qk + mask
        }
        qk <- qk$to(dtype = Rtorch::torch_float32)
        w <- Rtorch::nnf_softmax(qk, dim = - 1)$to(dtype = q$dtype)

        # Output
        out <- Rtorch::torch_matmul(w, v)
        out$permute(c(1, 3, 2, 4))$contiguous()$view(c(batch_size, seq_len, D))
    }
)

#' FSMN Multi-Head Attention
#'
#' Multi-head attention with Frequency-domain Self-attention Memory Network
#'
#' @param n_state Hidden dimension
#' @param n_head Number of heads
#' @param kernel_size FSMN kernel size (default 31)
#' @return nn_module
fsmn_multi_head_attention <- Rtorch::nn_module(
    "FSMNMultiHeadAttention",

    initialize = function (n_state, n_head, kernel_size = 31L)
    {
        self$n_head <- n_head
        self$n_state <- n_state
        self$head_dim <- n_state %/% n_head

        # Standard attention projections
        self$query <- Rtorch::nn_linear(n_state, n_state)
        self$key <- Rtorch::nn_linear(n_state, n_state, bias = FALSE)
        self$value <- Rtorch::nn_linear(n_state, n_state)
        self$out <- Rtorch::nn_linear(n_state, n_state)

        # FSMN block: depthwise conv for temporal context
        self$fsmn_block <- Rtorch::nn_conv1d(
            n_state, n_state, kernel_size,
            stride = 1L, padding = 0L, groups = n_state, bias = FALSE
        )
        self$left_padding <- (kernel_size - 1L) %/% 2L
        self$right_padding <- kernel_size - 1L - self$left_padding
    },

    forward_fsmn = function (inputs, mask = NULL)
    {
        # inputs: (batch, time, heads, head_dim)
        b <- inputs$size(1)
        t <- inputs$size(2)

        # Flatten heads
        inputs <- inputs$view(c(b, t, - 1))

        # Apply mask
        if (!is.null(mask) && mask$size(3) > 0) {
            inputs <- inputs * mask
        }

        # FSMN convolution: (B, T, C) -> (B, C, T) -> conv -> (B, C, T) -> (B, T, C)
        x <- inputs$transpose(2, 3)

        # Pad
        x <- Rtorch::nnf_pad(x, c(self$left_padding, self$right_padding))

        # Apply depthwise conv
        x <- self$fsmn_block$forward(x)
        x <- x$transpose(2, 3)

        # Residual connection
        x <- x + inputs

        if (!is.null(mask)) {
            x * mask
        } else {
            x
        }
    },

    qkv_attention = function (q, k, v, mask = NULL, mask_pad = NULL,
                              freqs_cis = NULL)
    {
        batch_size <- q$size(1)
        seq_len <- q$size(2)
        D <- q$size(3)
        scale <- (D / self$n_head) ^ (- 0.25)

        # Reshape
        q <- q$view(c(batch_size, seq_len, self$n_head, - 1))
        k <- k$view(c(batch_size, seq_len, self$n_head, - 1))
        v <- v$view(c(batch_size, seq_len, self$n_head, - 1))

        # Apply rotary embeddings if provided
        if (!is.null(freqs_cis)) {
            rotated <- apply_rotary_emb_s3(q, k, freqs_cis)
            q <- rotated$q
            k <- rotated$k
        }

        # FSMN on values
        fsm_memory <- self$forward_fsmn(v, mask_pad)

        # Attention computation
        q <- q$permute(c(1, 3, 2, 4)) * scale
        k <- k$permute(c(1, 3, 4, 2)) * scale
        v <- v$permute(c(1, 3, 2, 4))

        qk <- Rtorch::torch_matmul(q, k)
        if (!is.null(mask)) {
            qk <- qk + mask
        }
        qk <- qk$to(dtype = Rtorch::torch_float32)
        w <- Rtorch::nnf_softmax(qk, dim = - 1)$to(dtype = q$dtype)

        out <- Rtorch::torch_matmul(w, v)
        out <- out$permute(c(1, 3, 2, 4))$contiguous()$view(c(batch_size, seq_len, D))

        list(out = out, fsm_memory = fsm_memory)
    },

    forward = function (x, mask = NULL, mask_pad = NULL, freqs_cis = NULL)
    {
        q <- self$query$forward(x)
        k <- self$key$forward(x)
        v <- self$value$forward(x)

        result <- self$qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        self$out$forward(result$out) + result$fsm_memory
    }
)

# ============================================================================
# Residual Attention Block
# ============================================================================

#' Residual attention block
#'
#' @param n_state Hidden dimension
#' @param n_head Number of heads
#' @param kernel_size FSMN kernel size
#' @return nn_module
s3_residual_attention_block <- Rtorch::nn_module(
    "S3ResidualAttentionBlock",

    initialize = function (n_state, n_head, kernel_size = 31L)
    {
        self$attn <- fsmn_multi_head_attention(n_state, n_head, kernel_size)
        self$attn_ln <- Rtorch::nn_layer_norm(n_state, eps = 1e-6)

        n_mlp <- n_state * 4L
        self$mlp <- Rtorch::nn_sequential(
            Rtorch::nn_linear(n_state, n_mlp),
            Rtorch::nn_gelu(),
            Rtorch::nn_linear(n_mlp, n_state)
        )
        self$mlp_ln <- Rtorch::nn_layer_norm(n_state)
    },

    forward = function (x, mask = NULL, mask_pad = NULL, freqs_cis = NULL)
    {
        # Attention with pre-norm
        x <- x + self$attn$forward(self$attn_ln$forward(x), mask, mask_pad, freqs_cis)

        # MLP with pre-norm
        x <- x + self$mlp$forward(self$mlp_ln$forward(x))

        x
    }
)

# ============================================================================
# Audio Encoder V2
# ============================================================================

#' S3 Audio Encoder V2
#'
#' @param n_mels Number of mel bins
#' @param n_state Hidden dimension
#' @param n_head Number of attention heads
#' @param n_layer Number of transformer layers
#' @param stride Convolution stride (default 2)
#' @return nn_module
s3_audio_encoder <- Rtorch::nn_module(
    "S3AudioEncoderV2",

    initialize = function (n_mels, n_state, n_head, n_layer, stride = 2L)
    {
        self$stride <- stride

        # Two strided convolutions for downsampling
        self$conv1 <- Rtorch::nn_conv1d(n_mels, n_state, kernel_size = 3L,
            stride = stride, padding = 1L)
        self$conv2 <- Rtorch::nn_conv1d(n_state, n_state, kernel_size = 3L,
            stride = 2L, padding = 1L)

        # Precompute rotary embeddings
        self$freqs_cis <- precompute_freqs_cis(64L, 2048L)

        # Transformer blocks
        self$blocks <- Rtorch::nn_module_list(
            lapply(seq_len(n_layer), function (i)
            {
                    s3_residual_attention_block(n_state, n_head)
                })
        )
    },

    forward = function (x, x_len)
    {
        # x: (batch, n_mels, T)
        # x_len: (batch,)

        device <- x$device
        batch_size <- x$size(1)
        T <- x$size(3)

        # First conv with masking
        mask <- make_non_pad_mask_s3(x_len, T)$unsqueeze(2)$to(dtype = x$dtype, device = device)
        x <- Rtorch::nnf_gelu(self$conv1(x * mask))

        # Update lengths after first conv
        x_len <- Rtorch::torch_floor((x_len + 2 - 1 * (3 - 1) - 1) / self$stride + 1)$to(dtype = Rtorch::torch_long)
        x_slen <- as.integer((T + 2 - 1 * (3 - 1) - 1) / self$stride + 1)

        # Second conv with masking
        mask <- make_non_pad_mask_s3(x_len, x_slen)$unsqueeze(2)$to(dtype = x$dtype, device = device)
        x <- Rtorch::nnf_gelu(self$conv2(x * mask))

        # Update lengths after second conv
        x_len <- Rtorch::torch_floor((x_len + 2 - 1 * (3 - 1) - 1) / 2 + 1)$to(dtype = Rtorch::torch_long)
        x_slen <- as.integer((x_slen + 2 - 1 * (3 - 1) - 1) / self$stride + 1)

        # Create masks for attention
        mask <- make_non_pad_mask_s3(x_len, x_slen)$unsqueeze(2)$to(dtype = x$dtype, device = device)

        # Transpose to (batch, time, channels)
        x <- x$permute(c(1, 3, 2))

        # Prepare rotary embeddings
        freqs_cis <- self$freqs_cis$to(device = device)

        # Mask for attention bias
        mask_pad <- mask$transpose(2, 3)
        attn_mask <- mask_to_bias(mask, x$dtype)

        # Process through transformer blocks
        seq_len <- x$size(2)
        for (i in seq_along(self$blocks)) {
            block <- self$blocks[[i]]
            x <- block$forward(x, attn_mask$unsqueeze(2), mask_pad, freqs_cis[1:seq_len,])
        }

        list(hidden = x, lengths = x_len)
    }
)

# ============================================================================
# S3Tokenizer V2 (Main Module)
# ============================================================================

#' S3Tokenizer V2 module
#'
#' @param config Configuration list (default from s3_tokenizer_config())
#' @return nn_module
#' @export
s3_tokenizer <- Rtorch::nn_module(
    "S3TokenizerV2",

    initialize = function (config = NULL)
    {
        if (is.null(config)) {
            config <- s3_tokenizer_config()
        }
        self$config <- config

        # Create mel filterbank
        mel_fb <- create_mel_filterbank(
            sr = S3_SR,
            n_fft = 400L,
            n_mels = config$n_mels,
            fmin = 0,
            fmax = S3_SR / 2
        )
        self$mel_filters <- Rtorch::nn_buffer(
            Rtorch::torch_tensor(mel_fb, dtype = Rtorch::torch_float32)
        )

        # Hann window
        self$window <- Rtorch::nn_buffer(
            Rtorch::torch_hann_window(400L)
        )

        # Audio encoder
        self$encoder <- s3_audio_encoder(
            config$n_mels,
            config$n_audio_state,
            config$n_audio_head,
            config$n_audio_layer,
            stride = 2L
        )

        # Vector quantizer
        self$quantizer <- fsq_vector_quantization(
            config$n_audio_state,
            config$n_codebook_size
        )
    },

    log_mel_spectrogram = function (audio)
    {
        s3_log_mel_spectrogram(
            audio,
            self$mel_filters,
            self$window,
            n_fft = 400L,
            device = self$mel_filters$device
        )
    },

    quantize = function (mel, mel_len)
    {
        # mel: (batch, n_mels, T)
        # mel_len: (batch,)

        # Encode
        enc_result <- self$encoder$forward(mel, mel_len)
        hidden <- enc_result$hidden
        code_len <- enc_result$lengths

        # Quantize
        codes <- self$quantizer$encode(hidden)

        list(tokens = codes, lens = code_len)
    },

    forward = function (wavs, max_len = NULL)
    {
        device <- self$mel_filters$device

        # Handle input
        if (!inherits(wavs, "torch_tensor")) {
            wavs <- Rtorch::torch_tensor(wavs, dtype = Rtorch::torch_float32)
        }

        if (wavs$dim() == 1) {
            wavs <- wavs$unsqueeze(1)
        }

        wavs <- wavs$to(device = device)

        # Compute mel spectrogram
        mel <- self$log_mel_spectrogram(wavs)

        # Truncate if needed
        if (!is.null(max_len)) {
            mel <- mel[,, 1:min(mel$size(3), max_len * 4)]
        }

        # Get lengths
        mel_lens <- Rtorch::torch_tensor(mel$size(3), device = device)$unsqueeze(1)

        # Quantize
        result <- self$quantize(mel, mel_lens)

        list(
            tokens = result$tokens,
            lens = result$lens
        )
    }
)

# ============================================================================
# Helper Functions
# ============================================================================

#' Drop invalid speech tokens
#'
#' @param tokens Token tensor
#' @return Filtered tokens
drop_invalid_tokens <- function (tokens)
{
    # Remove tokens >= SPEECH_VOCAB_SIZE
    if (inherits(tokens, "torch_tensor")) {
        valid_mask <- tokens < SPEECH_VOCAB_SIZE
        tokens[valid_mask]
    } else {
        tokens[tokens < SPEECH_VOCAB_SIZE]
    }
}

#' Pad audio to multiple of token rate
#'
#' @param wav Audio samples
#' @param sr Sample rate
#' @return Padded audio
pad_audio_for_tokenizer <- function (wav, sr)
{
    if (!inherits(wav, "torch_tensor")) {
        wav <- Rtorch::torch_tensor(wav, dtype = Rtorch::torch_float32)
    }

    if (wav$dim() == 1) {
        wav <- wav$unsqueeze(1)
    }

    # Calculate intended length
    n_tokens <- ceiling(wav$size(2) / sr * S3_TOKEN_RATE)
    intended_len <- as.integer(n_tokens * sr / S3_TOKEN_RATE)

    # Pad if needed
    if (wav$size(2) < intended_len) {
        wav <- Rtorch::nnf_pad(wav, c(0L, intended_len - wav$size(2)), value = 0)
    }

    wav
}

# ============================================================================
# Weight Loading
# ============================================================================

#' Load S3Tokenizer weights from state dictionary
#'
#' @param model S3Tokenizer model
#' @param state_dict Named list of tensors
#' @param prefix Prefix for weight keys (default "tokenizer.")
#' @return Model with loaded weights
#' @export
load_s3tokenizer_weights <- function (model, state_dict, prefix = "tokenizer.")
{
    # Helper to copy weight if exists
    copy_if_exists <- function (r_param, key)
    {
        full_key <- paste0(prefix, key)
        if (full_key %in% names(state_dict)) {
            r_param$copy_(state_dict[[full_key]])
            return(TRUE)
        }
        # Try without prefix
        if (key %in% names(state_dict)) {
            r_param$copy_(state_dict[[key]])
            return(TRUE)
        }
        FALSE
    }

    Rtorch::with_no_grad({
            # Load mel filters and window (buffers)
            if (paste0(prefix, "_mel_filters") %in% names(state_dict)) {
                model$mel_filters$copy_(state_dict[[paste0(prefix, "_mel_filters")]])
            }
            if (paste0(prefix, "window") %in% names(state_dict)) {
                model$window$copy_(state_dict[[paste0(prefix, "window")]])
            }

            # Load encoder conv layers
            copy_if_exists(model$encoder$conv1$weight, "encoder.conv1.weight")
            copy_if_exists(model$encoder$conv1$bias, "encoder.conv1.bias")
            copy_if_exists(model$encoder$conv2$weight, "encoder.conv2.weight")
            copy_if_exists(model$encoder$conv2$bias, "encoder.conv2.bias")

            # Load transformer blocks
            for (i in seq_along(model$encoder$blocks)) {
                block_prefix <- sprintf("encoder.blocks.%d.", i - 1)
                block <- model$encoder$blocks[[i]]

                # Attention layer
                copy_if_exists(block$attn$query$weight, paste0(block_prefix, "attn.query.weight"))
                copy_if_exists(block$attn$query$bias, paste0(block_prefix, "attn.query.bias"))
                copy_if_exists(block$attn$key$weight, paste0(block_prefix, "attn.key.weight"))
                copy_if_exists(block$attn$value$weight, paste0(block_prefix, "attn.value.weight"))
                copy_if_exists(block$attn$value$bias, paste0(block_prefix, "attn.value.bias"))
                copy_if_exists(block$attn$out$weight, paste0(block_prefix, "attn.out.weight"))
                copy_if_exists(block$attn$out$bias, paste0(block_prefix, "attn.out.bias"))

                # FSMN block
                copy_if_exists(block$attn$fsmn_block$weight, paste0(block_prefix, "attn.fsmn_block.weight"))

                # Layer norms
                copy_if_exists(block$attn_ln$weight, paste0(block_prefix, "attn_ln.weight"))
                copy_if_exists(block$attn_ln$bias, paste0(block_prefix, "attn_ln.bias"))
                copy_if_exists(block$mlp_ln$weight, paste0(block_prefix, "mlp_ln.weight"))
                copy_if_exists(block$mlp_ln$bias, paste0(block_prefix, "mlp_ln.bias"))

                # MLP layers (sequential with indices 1 and 3)
                copy_if_exists(block$mlp[[1]]$weight, paste0(block_prefix, "mlp.0.weight"))
                copy_if_exists(block$mlp[[1]]$bias, paste0(block_prefix, "mlp.0.bias"))
                copy_if_exists(block$mlp[[3]]$weight, paste0(block_prefix, "mlp.2.weight"))
                copy_if_exists(block$mlp[[3]]$bias, paste0(block_prefix, "mlp.2.bias"))
            }

            # Load quantizer
            copy_if_exists(model$quantizer$codebook$project_down$weight, "quantizer._codebook.project_down.weight")
            copy_if_exists(model$quantizer$codebook$project_down$bias, "quantizer._codebook.project_down.bias")
        })

    model
}

