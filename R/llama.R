# Llama model implementation for chatterbox
# A minimal Llama implementation compatible with HuggingFace weights

# SDPA via Rtorch (backed by ATen's at::scaled_dot_product_attention)
get_sdpa <- function() Rtorch::torch_scaled_dot_product_attention

# ============================================================================
# Configuration
# ============================================================================

#' Create Llama 520M configuration
#'
#' @return List with model configuration
llama_config_520m <- function ()
{
    list(
        vocab_size = 8, # Unused - custom embeddings
        max_position_embeddings = 131072,
        hidden_size = 1024,
        intermediate_size = 4096,
        num_hidden_layers = 30,
        num_attention_heads = 16,
        num_key_value_heads = 16,
        head_dim = 64,
        hidden_act = "silu",
        attention_bias = FALSE,
        attention_dropout = 0.0,
        mlp_bias = FALSE,
        rms_norm_eps = 1e-5,
        rope_theta = 500000.0,
        rope_scaling = list(
            factor = 8.0,
            high_freq_factor = 4.0,
            low_freq_factor = 1.0,
            original_max_position_embeddings = 8192,
            rope_type = "llama3"
        )
    )
}

# ============================================================================
# RMSNorm
# ============================================================================

#' RMS Normalization module
#'
#' @param hidden_size Dimension to normalize
#' @param eps Epsilon for numerical stability
#' @return nn_module
llama_rms_norm <- Rtorch::nn_module(
    "LlamaRMSNorm",

    initialize = function (hidden_size, eps = 1e-5)
    {
        self$eps <- eps
        self$weight <- Rtorch::nn_parameter(Rtorch::torch_ones(hidden_size))
    },

    forward = function (x)
    {
        input_dtype <- x$dtype
        x <- x$to(dtype = Rtorch::torch_float32)
        variance <- x$pow(2)$mean(dim = - 1, keepdim = TRUE)
        x <- x * Rtorch::torch_rsqrt(variance + self$eps)
        self$weight * x$to(dtype = input_dtype)
    }
)

# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

#' Compute rotary position embeddings frequencies
#'
#' @param dim Dimension of embeddings
#' @param max_seq_len Maximum sequence length
#' @param theta Base frequency
#' @param scaling Rope scaling configuration (optional)
#' @param device Device to create tensors on
#' @return List with cos and sin caches
compute_rope_frequencies <- function (dim, max_seq_len, theta = 500000.0,
                                      scaling = NULL, device = "cpu")
{
    # Compute inverse frequencies
    # R torch_arange is inclusive; use end= - 1 for Python-like behavior
    inv_freq <- 1.0 / (theta ^ (Rtorch::torch_arange(start = 0, end = dim - 1, step = 2, device = device)$to(dtype = Rtorch::torch_float32) / dim))

    # Apply Llama3-style scaling if specified
    if (!is.null(scaling) && scaling$rope_type == "llama3") {
        inv_freq <- apply_llama3_rope_scaling(inv_freq, scaling, dim)
    }

    # Compute position indices (0 to max_seq_len-1)
    t <- Rtorch::torch_arange(0, max_seq_len - 1, device = device)$to(dtype = Rtorch::torch_float32)

    # Outer product: (seq_len, dim/2)
    freqs <- Rtorch::torch_outer(t, inv_freq)

    # Create cos and sin caches: (seq_len, dim)
    emb <- Rtorch::torch_cat(list(freqs, freqs), dim = - 1)

    list(
        cos = emb$cos(),
        sin = emb$sin()
    )
}

#' Apply Llama3-style RoPE scaling
#'
#' @param inv_freq Inverse frequencies
#' @param scaling Scaling configuration
#' @param dim Dimension
#' @return Scaled inverse frequencies
apply_llama3_rope_scaling <- function (inv_freq, scaling, dim)
{
    factor <- scaling$factor
    low_freq_factor <- scaling$low_freq_factor
    high_freq_factor <- scaling$high_freq_factor
    old_context_len <- scaling$original_max_position_embeddings

    low_freq_wavelen <- old_context_len / low_freq_factor
    high_freq_wavelen <- old_context_len / high_freq_factor

    wavelen <- 2 * pi / inv_freq

    # Apply scaling based on wavelength
    device <- inv_freq$device
    new_inv_freq <- Rtorch::torch_zeros_like(inv_freq)

    for (i in seq_len(inv_freq$size(1))) {
        wl <- as.numeric(wavelen[i])
        if (wl < high_freq_wavelen) {
            # High frequency: no scaling
            new_inv_freq[i] <- inv_freq[i]
        } else if (wl > low_freq_wavelen) {
            # Low frequency: full scaling
            new_inv_freq[i] <- inv_freq[i] / factor
        } else {
            # Medium frequency: smooth interpolation
            smooth <- (old_context_len / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv_freq[i] <- (1 - smooth) * inv_freq[i] / factor + smooth * inv_freq[i]
        }
    }

    new_inv_freq
}

#' Rotate half of the tensor for RoPE
#'
#' @param x Input tensor
#' @return Rotated tensor
rotate_half <- function (x)
{
    x1 <- x[,,, 1:(x$size(4) %/% 2)]
    x2 <- x[,,, (x$size(4) %/% 2 + 1) :x$size(4)]
    Rtorch::torch_cat(list(- x2, x1), dim = - 1)
}

#' Apply rotary position embeddings to Q and K
#'
#' @param q Query tensor (batch, heads, seq, head_dim)
#' @param k Key tensor (batch, heads, seq, head_dim)
#' @param cos Cosine cache
#' @param sin Sine cache
#' @param position_ids Position indices
#' @return List with rotated q and k
apply_rotary_pos_emb <- function (q, k, cos, sin, position_ids)
{
    # Gather cos/sin for positions
    # R torch uses 1-based indexing for unsqueeze, so unsqueeze(3) inserts at position 3
    cos <- cos[position_ids$add(1L),]$unsqueeze(3) # (batch, seq, 1, dim)
    sin <- sin[position_ids$add(1L),]$unsqueeze(3)

    # Transpose to match q, k layout: (batch, seq, 1, dim) -> (batch, 1, seq, dim)
    cos <- cos$transpose(2, 3) # (batch, 1, seq, dim)
    sin <- sin$transpose(2, 3)

    q_embed <- (q * cos) + (rotate_half(q) * sin)
    k_embed <- (k * cos) + (rotate_half(k) * sin)

    list(q = q_embed, k = k_embed)
}

# ============================================================================
# Llama Attention
# ============================================================================

#' Llama attention module
#'
#' @param config Model configuration
#' @param layer_idx Layer index
#' @return nn_module
llama_attention <- Rtorch::nn_module(
    "LlamaAttention",

    initialize = function (config, layer_idx)
    {
        self$config <- config
        self$layer_idx <- layer_idx
        self$hidden_size <- config$hidden_size
        self$num_heads <- config$num_attention_heads
        self$head_dim <- config$head_dim
        self$num_key_value_heads <- config$num_key_value_heads
        self$num_key_value_groups <- self$num_heads %/% self$num_key_value_heads
        self$attention_dropout <- config$attention_dropout

        # Projections
        self$q_proj <- Rtorch::nn_linear(self$hidden_size, self$num_heads * self$head_dim, bias = config$attention_bias)
        self$k_proj <- Rtorch::nn_linear(self$hidden_size, self$num_key_value_heads * self$head_dim, bias = config$attention_bias)
        self$v_proj <- Rtorch::nn_linear(self$hidden_size, self$num_key_value_heads * self$head_dim, bias = config$attention_bias)
        self$o_proj <- Rtorch::nn_linear(self$num_heads * self$head_dim, self$hidden_size, bias = config$attention_bias)
    },

    forward = function (hidden_states, position_ids, rope_cos, rope_sin,
                        attention_mask = NULL, past_key_value = NULL)
    {
        bsz <- hidden_states$size(1)
        q_len <- hidden_states$size(2)

        # Project Q, K, V
        query_states <- self$q_proj$forward(hidden_states)
        key_states <- self$k_proj$forward(hidden_states)
        value_states <- self$v_proj$forward(hidden_states)

        # Reshape: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        query_states <- query_states$view(c(bsz, q_len, self$num_heads, self$head_dim))$transpose(2, 3)
        key_states <- key_states$view(c(bsz, q_len, self$num_key_value_heads, self$head_dim))$transpose(2, 3)
        value_states <- value_states$view(c(bsz, q_len, self$num_key_value_heads, self$head_dim))$transpose(2, 3)

        # Apply RoPE
        rotated <- apply_rotary_pos_emb(query_states, key_states, rope_cos, rope_sin, position_ids)
        query_states <- rotated$q
        key_states <- rotated$k

        # Handle KV cache
        if (!is.null(past_key_value)) {
            key_states <- Rtorch::torch_cat(list(past_key_value$k, key_states), dim = 3)
            value_states <- Rtorch::torch_cat(list(past_key_value$v, value_states), dim = 3)
        }

        new_past_key_value <- list(k = key_states, v = value_states)

        # Repeat K, V for grouped query attention
        if (self$num_key_value_groups > 1) {
            key_states <- key_states$`repeat`(c(1, self$num_key_value_groups, 1, 1))
            value_states <- value_states$`repeat`(c(1, self$num_key_value_groups, 1, 1))
        }

        # Scaled dot-product attention (fused kernel)
        dropout_p <- if (self$training) self$attention_dropout else 0.0
        attn_output <- get_sdpa()(
            query_states,
            key_states,
            value_states,
            attn_mask = if (!is.null(attention_mask)) attention_mask else list(),
            dropout_p = dropout_p,
            is_causal = FALSE  # We handle causality via attention_mask
        )

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output <- attn_output$transpose(2, 3)$contiguous()
        attn_output <- attn_output$view(c(bsz, q_len, self$hidden_size))

        # Output projection
        attn_output <- self$o_proj$forward(attn_output)

        list(
            hidden_states = attn_output,
            past_key_value = new_past_key_value,
            attn_weights = NULL  # SDPA doesn't return attention weights
        )
    }
)

# ============================================================================
# Llama MLP
# ============================================================================

#' Llama MLP module
#'
#' @param config Model configuration
#' @return nn_module
llama_mlp <- Rtorch::nn_module(
    "LlamaMLP",

    initialize = function (config)
    {
        self$hidden_size <- config$hidden_size
        self$intermediate_size <- config$intermediate_size

        self$gate_proj <- Rtorch::nn_linear(self$hidden_size, self$intermediate_size, bias = config$mlp_bias)
        self$up_proj <- Rtorch::nn_linear(self$hidden_size, self$intermediate_size, bias = config$mlp_bias)
        self$down_proj <- Rtorch::nn_linear(self$intermediate_size, self$hidden_size, bias = config$mlp_bias)

        # SiLU activation
        self$act_fn <- function (x) x * Rtorch::torch_sigmoid(x)
    },

    forward = function(x) {
        # SwiGLU: down(silu(gate(x)) * up(x))
        self$down_proj$forward(self$act_fn(self$gate_proj$forward(x)) * self$up_proj$forward(x))
    }
)

# ============================================================================
# Llama Decoder Layer
# ============================================================================

#' Llama decoder layer
#'
#' @param config Model configuration
#' @param layer_idx Layer index
#' @return nn_module
llama_decoder_layer <- Rtorch::nn_module(
    "LlamaDecoderLayer",

    initialize = function(
        config,
        layer_idx
    ) {
        self$self_attn <- llama_attention(config, layer_idx)
        self$mlp <- llama_mlp(config)
        self$input_layernorm <- llama_rms_norm(config$hidden_size, config$rms_norm_eps)
        self$post_attention_layernorm <- llama_rms_norm(config$hidden_size, config$rms_norm_eps)
    },

    forward = function(
        hidden_states,
        position_ids,
        rope_cos,
        rope_sin,
        attention_mask = NULL,
        past_key_value = NULL
    ) {
        residual <- hidden_states

        # Pre-norm
        hidden_states <- self$input_layernorm$forward(hidden_states)

        # Self attention
        attn_out <- self$self_attn$forward(
            hidden_states,
            position_ids,
            rope_cos,
            rope_sin,
            attention_mask,
            past_key_value
        )
        hidden_states <- residual + attn_out$hidden_states

        # MLP
        residual <- hidden_states
        hidden_states <- self$post_attention_layernorm$forward(hidden_states)
        hidden_states <- residual + self$mlp$forward(hidden_states)

        list(
            hidden_states = hidden_states,
            past_key_value = attn_out$past_key_value,
            attn_weights = attn_out$attn_weights
        )
    }
)

# ============================================================================
# Llama Model
# ============================================================================

#' Llama model (decoder only)
#'
#' @param config Model configuration (default: 520M)
#' @return nn_module
llama_model <- Rtorch::nn_module(
    "LlamaModel",

    initialize = function(config = NULL) {
        if (is.null(config)) {
            config <- llama_config_520m()
        }
        self$config <- config

        # Token embeddings (not used when inputs_embeds provided)
        self$embed_tokens <- Rtorch::nn_embedding(config$vocab_size, config$hidden_size)

        # Decoder layers
        self$layers <- Rtorch::nn_module_list(
            lapply(seq_len(config$num_hidden_layers) - 1, function(i) {
                    llama_decoder_layer(config, i)
                })
        )

        # Final norm
        self$norm <- llama_rms_norm(config$hidden_size, config$rms_norm_eps)

        # RoPE cache (will be computed on first forward)
        self$rope_cache <- NULL
    },

    .get_rope_cache = function(
        seq_len,
        device
    ) {
        if (is.null(self$rope_cache) ||
            self$rope_cache$cos$size(1) < seq_len ||
            self$rope_cache$cos$device$type != device$type) {
            self$rope_cache <- compute_rope_frequencies(
                dim = self$config$head_dim,
                max_seq_len = max(seq_len, 4096),
                theta = self$config$rope_theta,
                scaling = self$config$rope_scaling,
                device = device
            )
        }
        self$rope_cache
    },

    forward = function(
        input_ids = NULL,
        inputs_embeds = NULL,
        position_ids = NULL,
        attention_mask = NULL,
        past_key_values = NULL,
        use_cache = TRUE,
        output_hidden_states = FALSE,
        output_attentions = FALSE
    ) {
        # Get hidden states from embeddings
        if (!is.null(inputs_embeds)) {
            hidden_states <- inputs_embeds
        } else if (!is.null(input_ids)) {
            hidden_states <- self$embed_tokens$forward(input_ids)
        } else {
            stop("Either input_ids or inputs_embeds must be provided")
        }

        batch_size <- hidden_states$size(1)
        seq_length <- hidden_states$size(2)
        device <- hidden_states$device

        # Handle position IDs
        if (is.null(position_ids)) {
            if (!is.null(past_key_values)) {
                past_length <- past_key_values[[1]]$k$size(3)
            } else {
                past_length <- 0
            }
            # R torch_arange is inclusive, so use seq_length - 1 for end to get seq_length values
            position_ids <- Rtorch::torch_arange(past_length, past_length + seq_length - 1, device = device, dtype = Rtorch::torch_long)
            position_ids <- position_ids$unsqueeze(1)$expand(c(batch_size, - 1))
        }

        # Get RoPE cache
        rope_cache <- self$.get_rope_cache(
            as.integer(position_ids$max()) + 1,
            device
        )

        # Create causal mask if needed
        if (is.null(attention_mask)) {
            if (seq_length > 1) {
                # Create causal mask
                # Use -65504 instead of -Inf to support float16 (autocast)
                mask <- Rtorch::torch_full(c(seq_length, seq_length), -65504.0, device = device)
                mask <- Rtorch::torch_triu(mask, diagonal = 1)
                # Add past length
                if (!is.null(past_key_values)) {
                    past_len <- past_key_values[[1]]$k$size(3)
                    mask <- Rtorch::torch_cat(list(
                            Rtorch::torch_zeros(c(seq_length, past_len), device = device),
                            mask
                        ), dim = - 1)
                }
                attention_mask <- mask$unsqueeze(1)$unsqueeze(1)
            }
        }

        # Storage for outputs
        if (output_hidden_states) {
            all_hidden_states <- list()
        } else {
            all_hidden_states <- NULL
        }
        if (output_attentions) {
            all_attentions <- list()
        } else {
            all_attentions <- NULL
        }
        if (use_cache) {
            new_past_key_values <- list()
        } else {
            new_past_key_values <- NULL
        }

        # Process through layers
        for (i in seq_along(self$layers)) {
            layer <- self$layers[[i]]

            if (output_hidden_states) {
                all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
            }

            if (!is.null(past_key_values)) {
                past_kv <- past_key_values[[i]]
            } else {
                past_kv <- NULL
            }

            layer_out <- layer$forward(
                hidden_states,
                position_ids,
                rope_cache$cos,
                rope_cache$sin,
                attention_mask,
                past_kv
            )

            hidden_states <- layer_out$hidden_states

            if (use_cache) {
                new_past_key_values[[i]] <- layer_out$past_key_value
            }

            if (output_attentions) {
                all_attentions[[length(all_attentions) + 1]] <- layer_out$attn_weights
            }
        }

        # Final norm
        hidden_states <- self$norm$forward(hidden_states)

        if (output_hidden_states) {
            all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
        }

        list(
            last_hidden_state = hidden_states,
            past_key_values = new_past_key_values,
            hidden_states = all_hidden_states,
            attentions = all_attentions
        )
    }
)

# ============================================================================
# Weight Loading
# ============================================================================

#' Load weights from safetensors into Llama model
#'
#' @param model LlamaModel instance
#' @param state_dict Named list of tensors from safetensors
#' @param prefix Prefix to strip from weight names (default: "model.")
#' @return Model with loaded weights
load_llama_weights <- function(
    model,
    state_dict,
    prefix = "model."
) {
    Rtorch::with_no_grad({
            # Map HuggingFace weight names to our model
            for (name in names(state_dict)) {
                # Strip prefix
                key <- sub(paste0("^", prefix), "", name)

                # Parse the key to find the parameter
                param <- tryCatch({
                        parts <- strsplit(key, "\\.") [[1]]

                        if (parts[1] == "embed_tokens") {
                            model$embed_tokens$weight
                        } else if (parts[1] == "norm") {
                            model$norm$weight
                        } else if (parts[1] == "layers") {
                            layer_idx <- as.integer(parts[2]) + 1# R is 1-indexed
                            layer <- model$layers[[layer_idx]]

                            if (parts[3] == "self_attn") {
                                proj_name <- parts[4]
                                if (proj_name == "q_proj") {
                                    layer$self_attn$q_proj$weight
                                } else if (proj_name == "k_proj") {
                                    layer$self_attn$k_proj$weight
                                } else if (proj_name == "v_proj") {
                                    layer$self_attn$v_proj$weight
                                } else if (proj_name == "o_proj") {
                                    layer$self_attn$o_proj$weight
                                }
                            } else if (parts[3] == "mlp") {
                                proj_name <- parts[4]
                                if (proj_name == "gate_proj") {
                                    layer$mlp$gate_proj$weight
                                } else if (proj_name == "up_proj") {
                                    layer$mlp$up_proj$weight
                                } else if (proj_name == "down_proj") {
                                    layer$mlp$down_proj$weight
                                }
                            } else if (parts[3] == "input_layernorm") {
                                layer$input_layernorm$weight
                            } else if (parts[3] == "post_attention_layernorm") {
                                layer$post_attention_layernorm$weight
                            }
                        }
                    }, error = function(e) NULL)

                if (!is.null(param)) {
                    # Copy weights
                    param$copy_(state_dict[[name]])
                }
            }
        })

    model
}

