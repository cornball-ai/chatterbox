# S3Gen - Speech Token to Waveform Generator
# Flow matching decoder + HiFiGAN vocoder

# Constants
S3GEN_SR <- 24000# Output sample rate

# ============================================================================
# Utility Functions
# ============================================================================

#' Create padding mask
#'
#' @param lengths Sequence lengths
#' @param max_len Maximum length
#' @return Boolean mask (TRUE for padded positions)
make_pad_mask <- function (lengths, max_len = NULL)
{
    if (is.null(max_len)) {
        max_len <- max(as.integer(lengths$cpu()))
    }

    batch_size <- lengths$size(1)
    device <- lengths$device

    # Create range tensor (0 to max_len-1)
    # R torch_arange(0, n) is inclusive, so 0 to n-1 gives n values
    range_tensor <- torch::torch_arange(0, max_len - 1, device = device, dtype = torch::torch_long())$unsqueeze(1)

    # Compare with lengths
    lengths <- lengths$view(c(1, batch_size))
    range_tensor >= lengths
}

# ============================================================================
# Conformer Encoder (Simplified)
# ============================================================================

# Use the validated conformer encoder from conformer.R
# (The full implementation is in upsample_conformer_encoder_full)
upsample_conformer_encoder <- function (input_size = 512, output_size = 512,
                                        num_blocks = 6)
{
    upsample_conformer_encoder_full(
        input_size = input_size,
        output_size = output_size,
        num_blocks = num_blocks,
        num_up_blocks = 4L,
        n_head = 8L,
        n_ffn = 2048L,
        dropout_rate = 0.1
    )
}

# ============================================================================
# Conditional Flow Matching Decoder
# ============================================================================

# Building blocks for ConditionalDecoder

#' Sinusoidal positional embedding for timesteps
#' @param dim Output dimension
#' @return nn_module
sinusoidal_pos_emb <- torch::nn_module(
    "SinusoidalPosEmb",

    initialize = function (dim = 320L)
    {
        self$dim <- dim
    },

    forward = function (x, scale = 1000)
    {
        if (x$dim() < 1) {
            x <- x$unsqueeze(1)
        }
        device <- x$device
        half_dim <- self$dim %/% 2L

        # emb = exp(arange(half_dim) * -log(10000) / (half_dim - 1))
        emb <- torch::torch_exp(
            torch::torch_arange(0, half_dim - 1, device = device, dtype = torch::torch_float32()) *
            (- log(10000) / (half_dim - 1))
        )

        # emb = scale * x * emb
        emb <- scale * x$unsqueeze(2) * emb$unsqueeze(1)

        # cat(sin, cos)
        torch::torch_cat(list(emb$sin(), emb$cos()), dim = - 1L)
    }
)

#' Timestep embedding MLP
#' @param in_channels Input channels
#' @param time_embed_dim Output dimension
#' @return nn_module
timestep_embedding <- torch::nn_module(
    "TimestepEmbedding",

    initialize = function (in_channels = 320L, time_embed_dim = 1024L)
    {
        self$linear_1 <- torch::nn_linear(in_channels, time_embed_dim)
        self$act <- torch::nn_silu()
        self$linear_2 <- torch::nn_linear(time_embed_dim, time_embed_dim)
    },

    forward = function (sample)
    {
        sample <- self$linear_1$forward(sample)
        sample <- self$act$forward(sample)
        sample <- self$linear_2$forward(sample)
        sample
    }
)

#' Causal Conv1d - pads left only
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param kernel_size Kernel size
#' @param stride Stride (default 1)
#' @param dilation Dilation (default 1)
#' @return nn_module
causal_conv1d <- torch::nn_module(
    "CausalConv1d",

    initialize = function (in_channels, out_channels, kernel_size, stride = 1L,
                           dilation = 1L)
    {
        self$conv <- torch::nn_conv1d(
            in_channels, out_channels, kernel_size,
            stride = stride, padding = 0L, dilation = dilation
        )
        # Causal padding: (kernel_size - 1) * dilation on left, 0 on right
        self$causal_padding <- c((kernel_size - 1L) * dilation, 0L)
    },

    forward = function (x)
    {
        x <- torch::nnf_pad(x, self$causal_padding)
        self$conv$forward(x)
    }
)

#' Transpose layer for use in sequential
#' @return nn_module
transpose_layer <- torch::nn_module(
    "Transpose",

    forward = function (x)
    {
        x$transpose(2L, 3L)$contiguous()
    }
)

#' Mish activation
#' @return nn_module
mish_activation <- torch::nn_module(
    "Mish",

    forward = function (x)
    {
        x * torch::torch_tanh(torch::nnf_softplus(x))
    }
)

#' Causal Block 1D - CausalConv + LayerNorm + Mish
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param kernel_size Kernel size
#' @return nn_module
causal_block1d <- torch::nn_module(
    "CausalBlock1D",

    initialize = function (in_channels, out_channels, kernel_size = 3L)
    {
        self$conv <- causal_conv1d(in_channels, out_channels, kernel_size)
        self$norm <- torch::nn_layer_norm(out_channels)
        self$mish <- mish_activation()
    },

    forward = function (x, mask)
    {
        # x: (B, C, T)
        h <- self$conv$forward(x * mask)
        # Transpose for layer norm
        h <- h$transpose(2L, 3L) # (B, T, C)
        h <- self$norm$forward(h)
        h <- h$transpose(2L, 3L) # (B, C, T)
        h <- self$mish$forward(h)
        h * mask
    }
)

#' Causal ResNet Block 1D
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param time_embed_dim Time embedding dimension
#' @return nn_module
causal_resnet_block1d <- torch::nn_module(
    "CausalResnetBlock1D",

    initialize = function (in_channels, out_channels, time_embed_dim = 1024L)
    {
        # Time embedding projection: Mish -> Linear
        self$mlp <- torch::nn_sequential(
            mish_activation(),
            torch::nn_linear(time_embed_dim, out_channels)
        )

        # Two causal blocks
        self$block1 <- causal_block1d(in_channels, out_channels)
        self$block2 <- causal_block1d(out_channels, out_channels)

        # Residual projection (1x1 conv always - Python has res_conv even for same channels)
        self$res_conv <- torch::nn_conv1d(in_channels, out_channels, 1L)
    },

    forward = function (x, mask, time_emb)
    {
        # x: (B, C, T), time_emb: (B, time_embed_dim)
        h <- self$block1$forward(x, mask)
        h <- h + self$mlp$forward(time_emb)$unsqueeze(- 1L)
        h <- self$block2$forward(h, mask)
        h + self$res_conv$forward(x * mask)
    }
)

#' Self-attention for transformer block
#' @param dim Hidden dimension
#' @param num_heads Number of attention heads
#' @param head_dim Head dimension (default 64)
#' @return nn_module
cfm_attention <- torch::nn_module(
    "CFMAttention",

    initialize = function (dim, num_heads = 8L, head_dim = 64L)
    {
        self$heads <- num_heads
        self$head_dim <- head_dim
        self$inner_dim <- num_heads * head_dim# 512 for default
        self$scale <- head_dim ^ (- 0.5)

        # Project from dim to inner_dim (256 -> 512)
        self$to_q <- torch::nn_linear(dim, self$inner_dim, bias = FALSE)
        self$to_k <- torch::nn_linear(dim, self$inner_dim, bias = FALSE)
        self$to_v <- torch::nn_linear(dim, self$inner_dim, bias = FALSE)

        # to_out is a ModuleList in Python (with dropout after linear)
        # For simplicity, just use linear (no dropout in inference)
        self$to_out <- torch::nn_module_list(list(
                torch::nn_linear(self$inner_dim, dim)
            ))
    },

    forward = function (hidden_states, attention_mask = NULL)
    {
        # hidden_states: (B, T, C)
        batch_size <- hidden_states$size(1)
        seq_len <- hidden_states$size(2)

        q <- self$to_q$forward(hidden_states)
        k <- self$to_k$forward(hidden_states)
        v <- self$to_v$forward(hidden_states)

        # Reshape to (B, heads, T, head_dim)
        q <- q$view(c(batch_size, seq_len, self$heads, self$head_dim))$transpose(2L, 3L)
        k <- k$view(c(batch_size, seq_len, self$heads, self$head_dim))$transpose(2L, 3L)
        v <- v$view(c(batch_size, seq_len, self$heads, self$head_dim))$transpose(2L, 3L)

        # Attention scores
        scores <- torch::torch_matmul(q, k$transpose(- 2L, - 1L)) * self$scale

        # Apply mask if provided
        if (!is.null(attention_mask)) {
            scores <- scores + attention_mask
        }

        attn <- torch::nnf_softmax(scores, dim = - 1L)
        out <- torch::torch_matmul(attn, v)

        # Reshape back to (B, T, inner_dim)
        out <- out$transpose(2L, 3L)$contiguous()$view(c(batch_size, seq_len, - 1L))
        # to_out is a ModuleList, first element is the linear projection
        self$to_out[[1]]$forward(out)
    }
)

#' GELU activation with projection (matches diffusers GELU structure)
#' @param dim_in Input dimension
#' @param dim_out Output dimension
#' @return nn_module
gelu_with_proj <- torch::nn_module(
    "GELUWithProj",

    initialize = function (dim_in, dim_out)
    {
        self$proj <- torch::nn_linear(dim_in, dim_out)
    },

    forward = function (x)
    {
        x <- self$proj$forward(x)
        torch::nnf_gelu(x, approximate = "tanh")
    }
)

#' Feed-forward network for transformer
#' Matches diffusers FeedForward: net = [GELU(proj), Dropout, Linear]
#' @param dim Input dimension
#' @param hidden_dim Hidden dimension (typically 4x dim)
#' @return nn_module
feed_forward <- torch::nn_module(
    "FeedForward",

    initialize = function (dim, hidden_dim = NULL)
    {
        if (is.null(hidden_dim)) {
            hidden_dim <- dim * 4L
        }
        # net[0]: GELU with projection (dim -> hidden_dim)
        # net[1]: Dropout (skipped in inference)
        # net[2]: Linear (hidden_dim -> dim)
        self$net <- torch::nn_module_list(list(
                gelu_with_proj(dim, hidden_dim),
                torch::nn_identity(), # Dropout placeholder
                torch::nn_linear(hidden_dim, dim)
            ))
    },

    forward = function (x)
    {
        for (i in seq_along(self$net)) {
            x <- self$net[[i]]$forward(x)
        }
        x
    }
)

#' Basic transformer block
#' @param dim Hidden dimension
#' @param num_heads Number of attention heads
#' @return nn_module
basic_transformer_block <- torch::nn_module(
    "BasicTransformerBlock",

    initialize = function (dim, num_heads = 8L)
    {
        self$norm1 <- torch::nn_layer_norm(dim)
        self$attn1 <- cfm_attention(dim, num_heads)
        self$norm3 <- torch::nn_layer_norm(dim)
        self$ff <- feed_forward(dim)
    },

    forward = function (hidden_states, attention_mask = NULL, timestep = NULL)
    {
        # Pre-norm self-attention
        norm_hidden <- self$norm1$forward(hidden_states)
        attn_out <- self$attn1$forward(norm_hidden, attention_mask)
        hidden_states <- hidden_states + attn_out

        # Pre-norm feed-forward
        norm_hidden <- self$norm3$forward(hidden_states)
        ff_out <- self$ff$forward(norm_hidden)
        hidden_states <- hidden_states + ff_out

        hidden_states
    }
)

#' CFM Estimator (ConditionalDecoder)
#'
#' UNet-style architecture with:
#' - 1 down block (320 -> 256, with 4 transformer blocks)
#' - 12 mid blocks (256 -> 256, each with 4 transformer blocks)
#' - 1 up block (512 -> 256 with skip connection, 4 transformer blocks)
#'
#' @param in_channels Input channels (default 320 = x + mu + spks + cond)
#' @param out_channels Output channels (default 80 = mel bins)
#' @param hidden_dim Hidden dimension (default 256)
#' @param num_mid_blocks Number of mid blocks (default 12)
#' @param num_transformer_blocks Transformer blocks per layer (default 4)
#' @return nn_module
cfm_estimator <- torch::nn_module(
    "CFMEstimator",

    initialize = function (in_channels = 320L, out_channels = 80L,
                           hidden_dim = 256L, num_mid_blocks = 12L,
                           num_transformer_blocks = 4L)
    {
        self$static_chunk_size <- 0L# For attention mask

        # Time embeddings: sinusoidal -> MLP
        self$time_embeddings <- sinusoidal_pos_emb(320L)
        self$time_mlp <- timestep_embedding(320L, 1024L)

        # Down block: resnet + 4 transformers + causal conv (no actual downsampling)
        self$down_resnet <- causal_resnet_block1d(in_channels, hidden_dim)
        self$down_transformers <- torch::nn_module_list(
            lapply(1:num_transformer_blocks, function (i) basic_transformer_block(hidden_dim))
        )
        self$down_conv <- causal_conv1d(hidden_dim, hidden_dim, 3L)

        # Mid blocks: 12 x (resnet + 4 transformers)
        self$mid_resnets <- torch::nn_module_list(
            lapply(1:num_mid_blocks, function(i) causal_resnet_block1d(hidden_dim, hidden_dim))
        )
        self$mid_transformers <- torch::nn_module_list(
            lapply(1:num_mid_blocks, function(i) {
                    torch::nn_module_list(
                        lapply(1:num_transformer_blocks, function(j) basic_transformer_block(hidden_dim))
                    )
                })
        )

        # Up block: causal conv + resnet (512->256) + 4 transformers
        self$up_conv <- causal_conv1d(hidden_dim, hidden_dim, 3L)
        self$up_resnet <- causal_resnet_block1d(hidden_dim * 2L, hidden_dim) # Skip connection doubles channels
        self$up_transformers <- torch::nn_module_list(
            lapply(1:num_transformer_blocks, function(i) basic_transformer_block(hidden_dim))
        )

        # Final block and projection
        self$final_block <- causal_block1d(hidden_dim, hidden_dim)
        self$final_proj <- torch::nn_conv1d(hidden_dim, out_channels, 1L)
    },

    forward = function(x, mask, mu, t, spks = NULL, cond = NULL) {
        # x: (B, 80, T) - noisy sample
        # mask: (B, 1, T) - padding mask
        # mu: (B, 80, T) - encoder output
        # t: (B,) - timestep
        # spks: (B, 80) - speaker embedding
        # cond: (B, 80, T) - conditioning (prompt mel)

        batch_size <- x$size(1)
        seq_len <- x$size(3)

        # Time embedding
        t_emb <- self$time_embeddings$forward(t)$to(dtype = t$dtype)
        t_emb <- self$time_mlp$forward(t_emb)

        # Pack inputs: x + mu + spks + cond -> (B, 320, T)
        h <- torch::torch_cat(list(x, mu), dim = 2L)
        if (!is.null(spks)) {
            spks_exp <- spks$unsqueeze(3L)$expand(c(- 1L, - 1L, seq_len))
            h <- torch::torch_cat(list(h, spks_exp), dim = 2L)
        }
        if (!is.null(cond)) {
            h <- torch::torch_cat(list(h, cond), dim = 2L)
        }

        # Down block
        h <- self$down_resnet$forward(h, mask, t_emb)
        h <- h$transpose(2L, 3L)$contiguous() # (B, T, C) for transformers
        attn_mask <- self$compute_attn_mask(h, mask)
        for (i in seq_along(self$down_transformers)) {
            h <- self$down_transformers[[i]]$forward(h, attn_mask, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous() # (B, C, T)
        hidden_skip <- h# Save for skip connection
        h <- self$down_conv$forward(h * mask)

        # Mid blocks
        for (i in seq_along(self$mid_resnets)) {
            h <- self$mid_resnets[[i]]$forward(h, mask, t_emb)
            h <- h$transpose(2L, 3L)$contiguous()
            for (j in seq_along(self$mid_transformers[[i]])) {
                h <- self$mid_transformers[[i]][[j]]$forward(h, attn_mask, t_emb)
            }
            h <- h$transpose(2L, 3L)$contiguous()
        }

        # Up block (Python order: concat -> resnet -> transformers -> conv)
        # Concat skip connection first
        h <- torch::torch_cat(list(h, hidden_skip), dim = 2L)
        # Resnet
        h <- self$up_resnet$forward(h, mask, t_emb)
        # Transformers
        h <- h$transpose(2L, 3L)$contiguous()
        for (i in seq_along(self$up_transformers)) {
            h <- self$up_transformers[[i]]$forward(h, attn_mask, t_emb)
        }
        h <- h$transpose(2L, 3L)$contiguous()
        # Up conv (after transformers, not before)
        h <- self$up_conv$forward(h * mask)

        # Final
        h <- self$final_block$forward(h, mask)
        h <- self$final_proj$forward(h * mask)

        h
    },

    compute_attn_mask = function(x, mask) {
        # For now, return NULL (no causal masking in inference)
        # The Python code uses chunk-based attention masks for streaming,
        # but for full sequence inference, we can use standard attention
        NULL
    }
)

#' Causal Conditional Flow Matching
#'
#' @param in_channels Input channels (x + mu + spks + cond)
#' @param out_channels Output channels (mel bins)
#' @param spk_emb_dim Speaker embedding dimension
#' @return nn_module
causal_cfm <- torch::nn_module(
    "CausalConditionalCFM",

    initialize = function(
        in_channels = 320,
        out_channels = 80,
        spk_emb_dim = 80
    ) {
        self$sigma_min <- 1e-6
        self$t_scheduler <- "cosine"
        self$inference_cfg_rate <- 0.7

        # Estimator network
        self$estimator <- cfm_estimator(in_channels, out_channels)

        # Pre-computed noise for reproducibility
        self$rand_noise <- torch::nn_buffer(
            torch::torch_randn(c(1, 80, 50 * 300))
        )
    },

    forward = function(
        mu,
        mask,
        spks,
        cond,
        n_timesteps = 10,
        temperature = 1.0
    ) {
        device <- mu$device
        seq_len <- mu$size(3)

        # Initial noise
        z <- self$rand_noise[,, 1:seq_len]$to(device = device)$to(dtype = mu$dtype) * temperature

        # Time span with cosine schedule
        t_span <- torch::torch_linspace(0, 1, n_timesteps + 1, device = device, dtype = mu$dtype)
        if (self$t_scheduler == "cosine") {
            t_span <- 1 - torch::torch_cos(t_span * 0.5 * pi)
        }

        # Euler solver
        result <- self$solve_euler(z, t_span, mu, mask, spks, cond)

        list(result, NULL) # Return mel and cache (NULL for now)
    },

    solve_euler = function(
        x,
        t_span,
        mu,
        mask,
        spks,
        cond
    ) {
        batch_size <- x$size(1)
        seq_len <- x$size(3)
        device <- x$device

        t <- t_span[1]$unsqueeze(1)
        dt <- t_span[2] - t_span[1]

        # Pre-allocate tensors for CFG (batch size 2)
        x_in <- torch::torch_zeros(c(2, 80, seq_len), device = device, dtype = x$dtype)
        mask_in <- torch::torch_zeros(c(2, 1, seq_len), device = device, dtype = x$dtype)
        mu_in <- torch::torch_zeros(c(2, 80, seq_len), device = device, dtype = x$dtype)
        t_in <- torch::torch_zeros(2, device = device, dtype = x$dtype)
        spks_in <- torch::torch_zeros(c(2, 80), device = device, dtype = x$dtype)
        cond_in <- torch::torch_zeros(c(2, 80, seq_len), device = device, dtype = x$dtype)

        for (step in 2:length(t_span)) {
            # Classifier-Free Guidance: conditional and unconditional paths
            x_in[1:2,,] <- x
            mask_in[1:2,,] <- mask
            mu_in[1,,] <- mu
            # mu_in[2] stays zero (unconditional)
            t_in[1:2] <- t
            spks_in[1,] <- spks
            # spks_in[2] stays zero
            cond_in[1,,] <- cond
            # cond_in[2] stays zero

            # Forward through estimator
            dphi_dt <- self$estimator$forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            # CFG combination
            dphi_cond <- dphi_dt[1,,]$unsqueeze(1)
            dphi_uncond <- dphi_dt[2,,]$unsqueeze(1)
            dphi_dt <- (1.0 + self$inference_cfg_rate) * dphi_cond - self$inference_cfg_rate * dphi_uncond

            # Euler step
            x <- x + dt * dphi_dt
            t <- t + dt

            if (step < length(t_span)) {
                dt <- t_span[step + 1] - t_span[step]
            }
        }

        x$to(dtype = torch::torch_float32())
    }
)

# ============================================================================
# S3Gen Flow Module
# ============================================================================

#' Causal Masked Diff with Xvector
#'
#' @param vocab_size Speech token vocabulary size
#' @param input_size Token embedding size
#' @param output_size Mel bins
#' @param spk_embed_dim Speaker embedding dimension
#' @param input_frame_rate Input frame rate for audio processing
#' @param token_mel_ratio Ratio of tokens to mel frames
#' @return nn_module
causal_masked_diff_xvec <- torch::nn_module(
    "CausalMaskedDiffWithXvec",

    initialize = function(
        vocab_size = 6561,
        input_size = 512,
        output_size = 80,
        spk_embed_dim = 192,
        input_frame_rate = 25,
        token_mel_ratio = 2
    ) {
        self$vocab_size <- vocab_size
        self$input_size <- input_size
        self$output_size <- output_size
        self$input_frame_rate <- input_frame_rate
        self$token_mel_ratio <- token_mel_ratio
        self$pre_lookahead_len <- 3

        # Token embedding
        self$input_embedding <- torch::nn_embedding(vocab_size, input_size)

        # Speaker embedding projection
        self$spk_embed_affine_layer <- torch::nn_linear(spk_embed_dim, output_size)

        # Encoder
        self$encoder <- upsample_conformer_encoder(input_size, input_size, 6)

        # Encoder output projection
        self$encoder_proj <- torch::nn_linear(input_size, output_size)

        # Flow matching decoder
        self$decoder <- causal_cfm(in_channels = 320, out_channels = output_size, spk_emb_dim = output_size)
    },

    forward = function(
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        finalize = TRUE
    ) {
        device <- token$device

        # Normalize and project speaker embedding
        embedding <- torch::nnf_normalize(embedding, dim = 2)
        embedding <- self$spk_embed_affine_layer$forward(embedding)

        # Concatenate prompt and speech tokens
        token <- torch::torch_cat(list(prompt_token, token), dim = 2)
        token_len <- prompt_token_len + token_len

        # Create mask
        mask <- (!make_pad_mask(token_len))$unsqueeze(3)$to(dtype = embedding$dtype, device = device)

        # Clamp tokens to valid range (ensure Long dtype preserved)
        token <- torch::torch_clamp(token, min = 0L, max = as.integer(self$vocab_size - 1))$to(dtype = torch::torch_long())

        # Embed tokens
        token <- self$input_embedding$forward(token$add(1L)) * mask# +1 for R indexing

        # Encode
        enc_result <- self$encoder$forward(token, token_len)
        h <- enc_result[[1]]
        h_lengths <- enc_result[[2]]

        # Truncate lookahead if not finalizing
        if (!finalize) {
            h <- h[, 1:(h$size(2) - self$pre_lookahead_len * self$token_mel_ratio),]
        }

        # Calculate mel lengths based on token counts (encoder upsamples by token_mel_ratio)
        prompt_token_len_scalar <- as.integer(prompt_token_len$cpu())
        mel_len1 <- prompt_token_len_scalar * self$token_mel_ratio
        mel_len2 <- as.integer(h$size(2)) - mel_len1

        # Project encoder output
        h <- self$encoder_proj$forward(h)

        # Prepare conditioning (resize prompt_feat to match expected mel_len1)
        conds <- torch::torch_zeros(c(1, mel_len1 + mel_len2, self$output_size),
            device = device, dtype = h$dtype)
        # Truncate or pad prompt_feat to mel_len1
        prompt_feat_len <- prompt_feat$size(2)
        if (prompt_feat_len >= mel_len1) {
            conds[1, 1:mel_len1,] <- prompt_feat[1, 1:mel_len1,]
        } else {
            conds[1, 1:prompt_feat_len,] <- prompt_feat
        }
        conds <- conds$transpose(2, 3)

        # Create mask for decoder
        dec_mask <- torch::torch_ones(c(1, 1, mel_len1 + mel_len2), device = device, dtype = h$dtype)

        # Run decoder
        h <- h$transpose(2, 3)
        result <- self$decoder$forward(
            mu = h,
            mask = dec_mask,
            spks = embedding$squeeze(2),
            cond = conds,
            n_timesteps = 10
        )
        feat <- result[[1]]

        # Extract generated portion (after prompt)
        feat <- feat[,, (mel_len1 + 1) :(mel_len1 + mel_len2)]

        list(feat$to(dtype = torch::torch_float32()), NULL)
    }
)

# ============================================================================
# S3Gen Token2Wav (Full Pipeline)
# ============================================================================

#' S3Gen Token to Waveform
#'
#' @return nn_module
s3gen <- torch::nn_module(
    "S3Gen",

    initialize = function() {
        # Speech tokenizer for reference audio (128 mels for S3TokenizerV2)
        self$tokenizer <- s3_tokenizer()

        # Mel spectrogram extractor
        # (reuse from audio_utils)

        # Speaker encoder (CAMPPlus)
        self$speaker_encoder <- campplus()

        # Flow matching decoder
        self$flow <- causal_masked_diff_xvec()

        # HiFiGAN vocoder (will be added)
        self$mel2wav <- NULL

        # Fade-in to reduce artifacts
        n_trim <- S3GEN_SR %/% 50# 20ms
        trim_fade <- torch::torch_zeros(2 * n_trim)
        fade_in <- (torch::torch_cos(torch::torch_linspace(pi, 0, n_trim)) + 1) / 2
        trim_fade[(n_trim + 1) :(2 * n_trim)] <- fade_in
        self$trim_fade <- torch::nn_buffer(trim_fade)
    },

#' Embed reference audio
    embed_ref = function(
        ref_wav,
        ref_sr,
        device = "auto"
    ) {
        if (device == "auto") {
            device <- self$tokenizer$mel_filters$device
        }

        # Convert to tensor
        if (!inherits(ref_wav, "torch_tensor")) {
            ref_wav <- torch::torch_tensor(ref_wav, dtype = torch::torch_float32())
        }

        if (ref_wav$dim() == 1) {
            ref_wav <- ref_wav$unsqueeze(1)
        }

        # Resample to 24kHz for mel extraction
        if (ref_sr != S3GEN_SR) {
            ref_wav_24 <- torch::torch_tensor(
                resample_audio(as.numeric(ref_wav$cpu()), ref_sr, S3GEN_SR),
                dtype = torch::torch_float32()
            )$unsqueeze(1)
        } else {
            ref_wav_24 <- ref_wav
        }

        # Compute mel spectrogram
        ref_mels <- compute_mel_spectrogram(ref_wav_24, sr = S3GEN_SR)
        ref_mels <- ref_mels$transpose(2, 3)$to(device = device)

        # Resample to 16kHz for speaker encoder and tokenizer
        ref_wav_16 <- torch::torch_tensor(
            resample_audio(as.numeric(ref_wav$cpu()), ref_sr, 16000),
            dtype = torch::torch_float32()
        )$unsqueeze(1)$to(device = device)

        # Speaker embedding (xvector)
        xvector <- compute_xvector_embedding(self$speaker_encoder, ref_wav_16, 16000)

        # Tokenize reference
        tok_result <- self$tokenizer$forward(ref_wav_16)
        ref_tokens <- tok_result$tokens
        ref_token_lens <- tok_result$lens

        list(
            prompt_token = ref_tokens$to(device = device),
            prompt_token_len = ref_token_lens,
            prompt_feat = ref_mels,
            prompt_feat_len = NULL,
            embedding = xvector
        )
    },

#' Run inference (tokens -> mel -> audio)
    inference = function(
        speech_tokens,
        ref_wav = NULL,
        ref_sr = NULL,
        ref_dict = NULL,
        finalize = TRUE
    ) {
        # Get reference dict
        if (is.null(ref_dict)) {
            if (is.null(ref_wav)) {
                stop("Must provide either ref_wav or ref_dict")
            }
            ref_dict <- self$embed_ref(ref_wav, ref_sr)
        }

        device <- ref_dict$embedding$device

        # Ensure tokens are 2D
        if (speech_tokens$dim() == 1) {
            speech_tokens <- speech_tokens$unsqueeze(1)
        }
        speech_tokens <- speech_tokens$to(device = device)
        speech_token_len <- torch::torch_tensor(speech_tokens$size(2), device = device)

        # Flow inference (tokens -> mel)
        result <- self$flow$forward(
            token = speech_tokens,
            token_len = speech_token_len,
            prompt_token = ref_dict$prompt_token,
            prompt_token_len = ref_dict$prompt_token_len,
            prompt_feat = ref_dict$prompt_feat,
            prompt_feat_len = ref_dict$prompt_feat_len,
            embedding = ref_dict$embedding,
            finalize = finalize
        )
        output_mels <- result[[1]]

        # Vocoder (mel -> audio)
        if (!is.null(self$mel2wav)) {
            vocoder_result <- self$mel2wav$inference(output_mels)
            output_wavs <- vocoder_result$audio

            # Apply fade-in
            fade_len <- length(self$trim_fade)
            output_wavs[, 1:fade_len] <- output_wavs[, 1:fade_len] * self$trim_fade

            return(list(output_wavs, NULL))
        }

        # Return mels if no vocoder
        list(output_mels, NULL)
    }
)

# ============================================================================
# Weight Loading
# ============================================================================

#' Load CFM estimator weights
#'
#' @param estimator CFM estimator module
#' @param state_dict State dictionary
#' @param prefix Key prefix in state dict
#' @return Number of keys loaded
#' @keywords internal
load_cfm_estimator_weights <- function(estimator, state_dict, prefix = "") {
    loaded <- 0L

    copy_if_exists <- function(r_param, key) {
        full_key <- paste0(prefix, key)
        if (full_key %in% names(state_dict)) {
            tryCatch({
                    torch::with_no_grad({
                            r_param$copy_(state_dict[[full_key]])
                        })
                    loaded <<- loaded + 1L
                    TRUE
                }, error = function(e) {
                    warning("Failed to copy ", full_key, ": ", e$message)
                    FALSE
                })
        } else {
            FALSE
        }
    }

    # Time MLP
    copy_if_exists(estimator$time_mlp$linear_1$weight, "time_mlp.linear_1.weight")
    copy_if_exists(estimator$time_mlp$linear_1$bias, "time_mlp.linear_1.bias")
    copy_if_exists(estimator$time_mlp$linear_2$weight, "time_mlp.linear_2.weight")
    copy_if_exists(estimator$time_mlp$linear_2$bias, "time_mlp.linear_2.bias")

    # Helper to load CausalBlock1D weights
    load_causal_block <- function(block, key_prefix) {
        # block.0 = CausalConv1d (conv inside)
        # block.2 = LayerNorm
        copy_if_exists(block$conv$conv$weight, paste0(key_prefix, "block.0.weight"))
        copy_if_exists(block$conv$conv$bias, paste0(key_prefix, "block.0.bias"))
        copy_if_exists(block$norm$weight, paste0(key_prefix, "block.2.weight"))
        copy_if_exists(block$norm$bias, paste0(key_prefix, "block.2.bias"))
    }

    # Helper to load CausalResnetBlock1D weights
    load_resnet_block <- function(resnet, key_prefix) {
        load_causal_block(resnet$block1, paste0(key_prefix, "block1."))
        load_causal_block(resnet$block2, paste0(key_prefix, "block2."))
        # mlp: [Mish, Linear] - mlp.1 is the linear
        copy_if_exists(resnet$mlp[[2]]$weight, paste0(key_prefix, "mlp.1.weight"))
        copy_if_exists(resnet$mlp[[2]]$bias, paste0(key_prefix, "mlp.1.bias"))
        # res_conv
        copy_if_exists(resnet$res_conv$weight, paste0(key_prefix, "res_conv.weight"))
        copy_if_exists(resnet$res_conv$bias, paste0(key_prefix, "res_conv.bias"))
    }

    # Helper to load BasicTransformerBlock weights
    load_transformer_block <- function(tfm, key_prefix) {
        # norm1
        copy_if_exists(tfm$norm1$weight, paste0(key_prefix, "norm1.weight"))
        copy_if_exists(tfm$norm1$bias, paste0(key_prefix, "norm1.bias"))
        # attn1
        copy_if_exists(tfm$attn1$to_q$weight, paste0(key_prefix, "attn1.to_q.weight"))
        copy_if_exists(tfm$attn1$to_k$weight, paste0(key_prefix, "attn1.to_k.weight"))
        copy_if_exists(tfm$attn1$to_v$weight, paste0(key_prefix, "attn1.to_v.weight"))
        copy_if_exists(tfm$attn1$to_out[[1]]$weight, paste0(key_prefix, "attn1.to_out.0.weight"))
        copy_if_exists(tfm$attn1$to_out[[1]]$bias, paste0(key_prefix, "attn1.to_out.0.bias"))
        # norm3
        copy_if_exists(tfm$norm3$weight, paste0(key_prefix, "norm3.weight"))
        copy_if_exists(tfm$norm3$bias, paste0(key_prefix, "norm3.bias"))
        # ff: net = [GELUWithProj, Dropout, Linear]
        copy_if_exists(tfm$ff$net[[1]]$proj$weight, paste0(key_prefix, "ff.net.0.proj.weight"))
        copy_if_exists(tfm$ff$net[[1]]$proj$bias, paste0(key_prefix, "ff.net.0.proj.bias"))
        copy_if_exists(tfm$ff$net[[3]]$weight, paste0(key_prefix, "ff.net.2.weight"))
        copy_if_exists(tfm$ff$net[[3]]$bias, paste0(key_prefix, "ff.net.2.bias"))
    }

    # Down block: down_blocks.0 = [resnet, [tfm0, tfm1, tfm2, tfm3], downsample]
    load_resnet_block(estimator$down_resnet, "down_blocks.0.0.")
    for (i in seq_along(estimator$down_transformers)) {
        load_transformer_block(estimator$down_transformers[[i]], paste0("down_blocks.0.1.", i - 1, "."))
    }
    copy_if_exists(estimator$down_conv$conv$weight, "down_blocks.0.2.weight")
    copy_if_exists(estimator$down_conv$conv$bias, "down_blocks.0.2.bias")

    # Mid blocks: mid_blocks.i = [resnet, [tfm0, tfm1, tfm2, tfm3]]
    for (i in seq_along(estimator$mid_resnets)) {
        load_resnet_block(estimator$mid_resnets[[i]], paste0("mid_blocks.", i - 1, ".0."))
        for (j in seq_along(estimator$mid_transformers[[i]])) {
            load_transformer_block(estimator$mid_transformers[[i]][[j]], paste0("mid_blocks.", i - 1, ".1.", j - 1, "."))
        }
    }

    # Up block: up_blocks.0 = [resnet, [tfm0, tfm1, tfm2, tfm3], upsample]
    load_resnet_block(estimator$up_resnet, "up_blocks.0.0.")
    for (i in seq_along(estimator$up_transformers)) {
        load_transformer_block(estimator$up_transformers[[i]], paste0("up_blocks.0.1.", i - 1, "."))
    }
    copy_if_exists(estimator$up_conv$conv$weight, "up_blocks.0.2.weight")
    copy_if_exists(estimator$up_conv$conv$bias, "up_blocks.0.2.bias")

    # Final block and projection
    copy_if_exists(estimator$final_block$conv$conv$weight, "final_block.block.0.weight")
    copy_if_exists(estimator$final_block$conv$conv$bias, "final_block.block.0.bias")
    copy_if_exists(estimator$final_block$norm$weight, "final_block.block.2.weight")
    copy_if_exists(estimator$final_block$norm$bias, "final_block.block.2.bias")
    copy_if_exists(estimator$final_proj$weight, "final_proj.weight")
    copy_if_exists(estimator$final_proj$bias, "final_proj.bias")

    loaded
}

#' Load S3Gen weights
#'
#' @param model S3Gen model
#' @param state_dict State dictionary from safetensors
#' @return Model with loaded weights
#' @export
load_s3gen_weights <- function(
    model,
    state_dict
) {
    torch::with_no_grad({
            # Helper to copy weight if exists
            copy_if_exists <- function(
                r_param,
                key
            ) {
                if (key %in% names(state_dict)) {
                    tryCatch({
                            r_param$copy_(state_dict[[key]])
                            return(TRUE)
                        }, error = function(e) {
                            warning("Failed to copy ", key, ": ", e$message)
                            return(FALSE)
                        })
                }
                FALSE
            }

            # ========== Speech Tokenizer ==========
            # Load tokenizer weights (S3TokenizerV2)
            load_s3tokenizer_weights(model$tokenizer, state_dict, prefix = "tokenizer.")

            # ========== Speaker Encoder (CAMPPlus) ==========
            load_campplus_weights(model$speaker_encoder, state_dict, prefix = "speaker_encoder.")

            # ========== Flow Module ==========
            # Input embedding
            copy_if_exists(model$flow$input_embedding$weight, "flow.input_embedding.weight")

            # Speaker embedding projection
            copy_if_exists(model$flow$spk_embed_affine_layer$weight, "flow.spk_embed_affine_layer.weight")
            copy_if_exists(model$flow$spk_embed_affine_layer$bias, "flow.spk_embed_affine_layer.bias")

            # Encoder projection
            copy_if_exists(model$flow$encoder_proj$weight, "flow.encoder_proj.weight")
            copy_if_exists(model$flow$encoder_proj$bias, "flow.encoder_proj.bias")

            # Encoder - use validated conformer encoder weight loading
            load_conformer_encoder_weights(model$flow$encoder, state_dict, prefix = "flow.encoder.")

            # CFM Decoder/Estimator - ConditionalDecoder (UNet-style)
            load_cfm_estimator_weights(model$flow$decoder$estimator, state_dict, prefix = "flow.decoder.estimator.")

            # ========== HiFiGAN Vocoder ==========
            if (!is.null(model$mel2wav)) {
                load_hifigan_weights(model$mel2wav, state_dict, prefix = "mel2wav.")
            }
        })

    model
}

#' Load S3Gen from safetensors file
#'
#' @param path Path to s3gen.safetensors
#' @param device Device to load to ("cpu", "cuda", etc.)
#' @return S3Gen model with loaded weights
#' @export
load_s3gen <- function(
    path,
    device = "cpu"
) {
    # Read safetensors
    state_dict <- read_safetensors(path, device)

    # Create model
    model <- s3gen()

    # Create and attach vocoder
    model$mel2wav <- create_s3gen_vocoder(device)

    # Load weights
    model <- load_s3gen_weights(model, state_dict)

    # Move to device and eval mode
    model$to(device = device)
    model$eval()

    model
}

