# Traced Llama modules for jit_trace optimization
# These modules use pre-allocated KV cache with masking for fixed tensor shapes

# ============================================================================
# Traceable KV Projector
# ============================================================================

#' Traceable K/V projection module
#'
#' Computes K and V projections with RoPE for a single layer.
#' Returns concatenated K and V for easy unpacking.
#'
#' @param layer Original llama_decoder_layer
#' @return nn_module
traceable_kv_projector <- Rtorch::nn_module(
    "TraceableKVProjector",

    initialize = function(layer) {
        self$input_layernorm <- layer$input_layernorm
        self$k_proj <- layer$self_attn$k_proj
        self$v_proj <- layer$self_attn$v_proj
        self$num_heads <- layer$self_attn$num_heads
        self$head_dim <- layer$self_attn$head_dim
    },

    forward = function(hidden_states, position_ids, rope_cos, rope_sin) {
        bsz <- hidden_states$size(1)

        # LayerNorm
        normed <- self$input_layernorm$forward(hidden_states)

        # K, V projections
        k <- self$k_proj$forward(normed)
        v <- self$v_proj$forward(normed)

        # Reshape
        k <- k$view(c(bsz, 1L, self$num_heads, self$head_dim))$transpose(2L, 3L)
        v <- v$view(c(bsz, 1L, self$num_heads, self$head_dim))$transpose(2L, 3L)

        # Apply RoPE to K
        rotated <- apply_rotary_pos_emb(k, k, rope_cos, rope_sin, position_ids)
        k <- rotated$k

        # Return concatenated K and V: (batch, heads, 1, head_dim * 2)
        Rtorch::torch_cat(list(k, v), dim = -1L)
    }
)

# ============================================================================
# Traceable Attention (Pre-allocated Cache)
# ============================================================================

#' Traceable attention module with pre-allocated KV cache
#'
#' This module is designed to be traced with jit_trace. It uses:
#' - Pre-allocated KV cache of fixed max size
#' - Attention mask to indicate valid cache positions
#' - Returns only output tensor (no lists/dicts)
#'
#' @param attn Original llama_attention module
#' @param max_cache_len Maximum cache length
#' @return nn_module
traceable_attention <- Rtorch::nn_module(
    "TraceableAttention",

    initialize = function(attn, max_cache_len = 300L) {
        # Copy projections from original attention
        self$q_proj <- attn$q_proj
        self$k_proj <- attn$k_proj
        self$v_proj <- attn$v_proj
        self$o_proj <- attn$o_proj

        self$num_heads <- attn$num_heads
        self$head_dim <- attn$head_dim
        self$hidden_size <- attn$hidden_size
        self$num_key_value_heads <- attn$num_key_value_heads
        self$num_key_value_groups <- attn$num_key_value_groups
        self$max_cache_len <- max_cache_len
    },

    forward = function(hidden_states, position_ids, rope_cos, rope_sin,
                       k_cache, v_cache, valid_mask) {
        # hidden_states: (batch, 1, hidden_size) - single token
        # k_cache, v_cache: (batch, heads, max_cache_len, head_dim) - pre-allocated
        # valid_mask: (batch, 1, 1, max_cache_len) - TRUE for valid positions

        bsz <- hidden_states$size(1)
        q_len <- hidden_states$size(2)

        # Project Q, K, V for current token
        query_states <- self$q_proj$forward(hidden_states)
        key_states <- self$k_proj$forward(hidden_states)
        value_states <- self$v_proj$forward(hidden_states)

        # Reshape
        query_states <- query_states$view(c(bsz, q_len, self$num_heads, self$head_dim))$transpose(2L, 3L)
        key_states <- key_states$view(c(bsz, q_len, self$num_key_value_heads, self$head_dim))$transpose(2L, 3L)
        value_states <- value_states$view(c(bsz, q_len, self$num_key_value_heads, self$head_dim))$transpose(2L, 3L)

        # Apply RoPE to current token
        rotated <- apply_rotary_pos_emb(query_states, key_states, rope_cos, rope_sin, position_ids)
        query_states <- rotated$q
        # Note: key_states and value_states are written to cache outside trace

        # Repeat K/V for grouped query attention
        if (self$num_key_value_groups > 1L) {
            k_cache <- k_cache$`repeat`(c(1L, self$num_key_value_groups, 1L, 1L))
            v_cache <- v_cache$`repeat`(c(1L, self$num_key_value_groups, 1L, 1L))
        }

        # Create attention mask from valid_mask
        # valid_mask is TRUE where positions are valid
        # We need 0 for valid, -inf for invalid
        attn_mask <- Rtorch::torch_where(
            valid_mask,
            Rtorch::torch_zeros(1L, device = query_states$device, dtype = query_states$dtype),
            Rtorch::torch_full(c(1L), -65504.0, device = query_states$device, dtype = query_states$dtype)
        )

        # SDPA with mask
        sdpa <- get_sdpa()
        attn_output <- sdpa(
            query_states,
            k_cache,
            v_cache,
            attn_mask = attn_mask,
            dropout_p = 0.0,
            is_causal = FALSE
        )

        # Reshape back
        attn_output <- attn_output$transpose(2L, 3L)$contiguous()
        attn_output <- attn_output$view(c(bsz, q_len, self$hidden_size))

        # Output projection
        self$o_proj$forward(attn_output)
    }
)

# ============================================================================
# Traceable Decoder Layer
# ============================================================================

#' Traceable decoder layer with pre-allocated KV cache
#'
#' @param layer Original llama_decoder_layer
#' @param max_cache_len Maximum cache length
#' @return nn_module
traceable_decoder_layer <- Rtorch::nn_module(
    "TraceableDecoderLayer",

    initialize = function(layer, max_cache_len = 300L) {
        self$self_attn <- traceable_attention(layer$self_attn, max_cache_len)
        self$mlp <- layer$mlp
        self$input_layernorm <- layer$input_layernorm
        self$post_attention_layernorm <- layer$post_attention_layernorm
    },

    forward = function(hidden_states, position_ids, rope_cos, rope_sin,
                       k_cache, v_cache, valid_mask) {
        residual <- hidden_states

        # Pre-norm
        hidden_states <- self$input_layernorm$forward(hidden_states)

        # Self attention with pre-allocated cache
        attn_out <- self$self_attn$forward(
            hidden_states, position_ids, rope_cos, rope_sin,
            k_cache, v_cache, valid_mask
        )
        hidden_states <- residual + attn_out

        # MLP
        residual <- hidden_states
        hidden_states <- self$post_attention_layernorm$forward(hidden_states)
        hidden_states <- residual + self$mlp$forward(hidden_states)

        hidden_states
    }
)

# ============================================================================
# Traceable Transformer (Full Model)
# ============================================================================

#' Traceable transformer for cached inference
#'
#' This wraps the full Llama model for traced cached inference.
#' Uses pre-allocated KV cache for all layers.
#'
#' @param tfmr Original llama_model
#' @param max_cache_len Maximum cache length
#' @return nn_module
traceable_transformer_cached <- Rtorch::nn_module(
    "TraceableTransformerCached",

    initialize = function(tfmr, max_cache_len = 300L) {
        self$n_layers <- length(tfmr$layers)
        self$layers <- Rtorch::nn_module_list(
            lapply(seq_len(self$n_layers), function(i) {
                traceable_decoder_layer(tfmr$layers[[i]], max_cache_len)
            })
        )
        self$norm <- tfmr$norm
        self$max_cache_len <- max_cache_len
    },

    forward = function(hidden_states, position_ids, rope_cos, rope_sin,
                       k_caches, v_caches, valid_mask) {
        # k_caches, v_caches: list of (batch, heads, max_len, head_dim) tensors
        # We pass them as a single stacked tensor for tracing: (n_layers, batch, heads, max_len, head_dim)

        for (i in seq_len(self$n_layers)) {
            hidden_states <- self$layers[[i]]$forward(
                hidden_states, position_ids, rope_cos, rope_sin,
                k_caches[i,,,,,drop=FALSE]$squeeze(1L),
                v_caches[i,,,,,drop=FALSE]$squeeze(1L),
                valid_mask
            )
        }

        self$norm$forward(hidden_states)
    }
)

# ============================================================================
# Traceable Transformer (First Token - No Cache)
# ============================================================================

#' Traceable transformer for first token (no cache)
#'
#' @param tfmr Original llama_model
#' @return nn_module
traceable_transformer_first <- Rtorch::nn_module(
    "TraceableTransformerFirst",

    initialize = function(tfmr) {
        self$n_layers <- length(tfmr$layers)
        self$layers <- tfmr$layers
        self$norm <- tfmr$norm
        self$config <- tfmr$config
    },

    forward = function(hidden_states, position_ids, rope_cos, rope_sin, attention_mask) {
        # First token forward - returns hidden states and K/V for cache
        # Note: For tracing, we need to return a single tensor
        # K/V cache values will be extracted separately

        for (i in seq_len(self$n_layers)) {
            result <- self$layers[[i]]$forward(
                hidden_states, position_ids, rope_cos, rope_sin,
                attention_mask, NULL  # No past KV for first token
            )
            hidden_states <- result$hidden_states
        }

        self$norm$forward(hidden_states)
    }
)

# ============================================================================
# KV Cache Manager
# ============================================================================

#' Create pre-allocated KV cache
#'
#' @param batch_size Batch size
#' @param n_layers Number of transformer layers
#' @param n_heads Number of attention heads
#' @param head_dim Head dimension
#' @param max_len Maximum sequence length
#' @param device Device to allocate on
#' @return List with k_cache, v_cache, valid_mask
create_kv_cache <- function(batch_size, n_layers, n_heads, head_dim, max_len, device) {
    # Stacked caches: (n_layers, batch, heads, max_len, head_dim)
    k_cache <- Rtorch::torch_zeros(
        c(n_layers, batch_size, n_heads, max_len, head_dim),
        device = device
    )
    v_cache <- Rtorch::torch_zeros(
        c(n_layers, batch_size, n_heads, max_len, head_dim),
        device = device
    )

    # Valid mask: (batch, 1, 1, max_len) - shared across layers
    valid_mask <- Rtorch::torch_zeros(
        c(batch_size, 1L, 1L, max_len),
        dtype = Rtorch::torch_bool,
        device = device
    )

    list(
        k_cache = k_cache,
        v_cache = v_cache,
        valid_mask = valid_mask
    )
}

#' Update KV cache with new K/V values
#'
#' @param cache Cache list from create_kv_cache
#' @param layer_idx Layer index (1-indexed)
#' @param new_k New key tensor (batch, heads, 1, head_dim)
#' @param new_v New value tensor (batch, heads, 1, head_dim)
#' @param position Current position (0-indexed)
update_kv_cache <- function(cache, layer_idx, new_k, new_v, position) {
    # Write to cache at current position
    # Position is 0-indexed, R tensors are 1-indexed
    pos_r <- position + 1L
    max_len <- cache$k_cache$size(4)

    if (pos_r > max_len) {
        stop(sprintf("KV cache position %d exceeds max length %d", pos_r, max_len))
    }

    cache$k_cache[layer_idx,,, pos_r, ] <- new_k$squeeze(3L)
    cache$v_cache[layer_idx,,, pos_r, ] <- new_v$squeeze(3L)

    invisible(cache)
}

#' Update valid mask to include new position
#'
#' @param cache Cache list from create_kv_cache
#' @param position Current position (0-indexed)
update_valid_mask <- function(cache, position) {
    pos_r <- position + 1L
    cache$valid_mask[,,, pos_r] <- TRUE
    invisible(cache)
}

#' Initialize cache with first token K/V values
#'
#' @param cache Cache list from create_kv_cache
#' @param past_key_values List of K/V from first forward pass
#' @return Updated cache (and seq_len as attribute)
init_cache_from_first <- function(cache, past_key_values) {
    n_layers <- length(past_key_values)
    seq_len <- past_key_values[[1]]$k$size(3)
    max_len <- cache$k_cache$size(4)

    # Check if seq_len fits in cache
    if (seq_len > max_len) {
        stop(sprintf("First token sequence length (%d) exceeds max cache length (%d). Increase max_cache_len.", seq_len, max_len))
    }

    for (i in seq_len(n_layers)) {
        kv <- past_key_values[[i]]
        # kv$k, kv$v are (batch, heads, seq_len, head_dim)
        cache$k_cache[i,,, 1:seq_len, ] <- kv$k
        cache$v_cache[i,,, 1:seq_len, ] <- kv$v
    }

    # Mark all first-token positions as valid
    cache$valid_mask[,,, 1:seq_len] <- TRUE

    # Store seq_len for later reference
    cache$current_len <- seq_len

    invisible(cache)
}
