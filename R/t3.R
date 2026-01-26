# T3 Model implementation for chatterbox
# Token-to-Token TTS model using Llama backbone

# ============================================================================
# Configuration
# ============================================================================

#' Create T3 configuration (English-only)
#'
#' @return List with T3 configuration
t3_config_english <- function ()
{
    list(
        start_text_token = 255,
        stop_text_token = 0,
        text_tokens_dict_size = 704,
        max_text_tokens = 2048,

        start_speech_token = 6561,
        stop_speech_token = 6562,
        speech_tokens_dict_size = 8194,
        max_speech_tokens = 4096,

        llama_config_name = "Llama_520M",
        input_pos_emb = "learned",
        speech_cond_prompt_len = 150,

        encoder_type = "voice_encoder",
        speaker_embed_size = 256,
        use_perceiver_resampler = TRUE,
        emotion_adv = TRUE,

        # Derived from Llama config
        n_channels = 1024
    )
}

# ============================================================================
# Learned Position Embeddings
# ============================================================================

#' Learned position embeddings module
#'
#' @param seq_len Maximum sequence length
#' @param model_dim Embedding dimension
#' @param init_std Initialization standard deviation
#' @return nn_module
learned_position_embeddings <- torch::nn_module(
    "LearnedPositionEmbeddings",

    initialize = function (seq_len, model_dim, init_std = 0.02)
    {
        self$emb <- torch::nn_embedding(seq_len, model_dim)
        # GPT-2 style initialization
        torch::with_no_grad({
                self$emb$weight$normal_(mean = 0.0, std = init_std)
            })
    },

    forward = function (x)
    {
        # Returns positional embeddings for indices 0 to length of x
        sl <- x$size(2)
        device <- x$device
        # R torch_arange(0, n) is inclusive, so 0 to sl-1 gives sl values
        indices <- torch::torch_arange(0, sl - 1, device = device, dtype = torch::torch_long())
        self$emb$forward(indices$add(1L)) # R/torch is 1-indexed for embeddings
    },

    get_fixed_embedding = function (idx)
    {
        # Get positional embedding for specific index/indices
        device <- self$emb$weight$device

        if (!inherits(idx, "torch_tensor")) {
            idx <- torch::torch_tensor(idx, device = device, dtype = torch::torch_long())
        } else {
            idx <- idx$to(device = device)
        }

        # Ensure at least 2D
        if (idx$dim() == 0) {
            idx <- idx$unsqueeze(1)$unsqueeze(1)
        } else if (idx$dim() == 1) {
            idx <- idx$unsqueeze(1)
        }

        self$emb$forward(idx$add(1L)) # (B, T, dim)
    }
)

# ============================================================================
# T3 Conditioning
# ============================================================================

#' Create T3 conditioning object
#'
#' @param speaker_emb Speaker embedding tensor (B, 256)
#' @param cond_prompt_speech_tokens Optional speech tokens for conditioning
#' @param cond_prompt_speech_emb Optional pre-computed speech embeddings
#' @param emotion_adv Emotion/exaggeration control (0-1)
#' @return List representing T3Cond
t3_cond <- function (speaker_emb, cond_prompt_speech_tokens = NULL,
                     cond_prompt_speech_emb = NULL, emotion_adv = 0.5)
{
    list(
        speaker_emb = speaker_emb,
        clap_emb = NULL, # Not implemented
        cond_prompt_speech_tokens = cond_prompt_speech_tokens,
        cond_prompt_speech_emb = cond_prompt_speech_emb,
        emotion_adv = emotion_adv
    )
}

#' Move T3 conditioning to device
#'
#' @param cond T3 conditioning object
#' @param device Target device
#' @return T3 conditioning on device
t3_cond_to_device <- function (cond, device)
{
    result <- cond
    for (name in names(cond)) {
        if (inherits(cond[[name]], "torch_tensor")) {
            result[[name]] <- cond[[name]]$to(device = device)
        }
    }
    result
}

# ============================================================================
# Attention Block for Perceiver (matches Python's AttentionBlock2)
# ============================================================================

#' Attention block for perceiver
#'
#' @param embed_dim Embedding dimension (default 1024)
#' @param num_heads Number of attention heads (default 4)
#' @return nn_module
attention_block <- torch::nn_module(
    "AttentionBlock",

    initialize = function (embed_dim = 1024, num_heads = 4)
    {
        self$embed_dim <- embed_dim
        self$num_heads <- num_heads
        self$head_dim <- embed_dim %/% num_heads

        # Single norm layer (applied to both inputs)
        self$norm <- torch::nn_layer_norm(embed_dim)

        # Q, K, V projections
        self$to_q <- torch::nn_linear(embed_dim, embed_dim)
        self$to_k <- torch::nn_linear(embed_dim, embed_dim)
        self$to_v <- torch::nn_linear(embed_dim, embed_dim)

        # Output projection
        self$proj_out <- torch::nn_linear(embed_dim, embed_dim)
    },

    forward = function (x1, x2)
    {
        # x1: query source (batch, q_len, dim)
        # x2: key/value source (batch, kv_len, dim)
        batch_size <- x1$size(1)
        q_len <- x1$size(2)
        kv_len <- x2$size(2)

        # Normalize both inputs
        x1_norm <- self$norm$forward(x1)
        x2_norm <- self$norm$forward(x2)

        # Create Q from x1, K and V from x2
        q <- self$to_q$forward(x1_norm)
        k <- self$to_k$forward(x2_norm)
        v <- self$to_v$forward(x2_norm)

        # Reshape to (batch, heads, seq, head_dim)
        q <- q$view(c(batch_size, q_len, self$num_heads, self$head_dim))$transpose(2, 3)
        k <- k$view(c(batch_size, kv_len, self$num_heads, self$head_dim))$transpose(2, 3)
        v <- v$view(c(batch_size, kv_len, self$num_heads, self$head_dim))$transpose(2, 3)

        # Scaled dot-product attention
        scale <- 1.0 / sqrt(self$head_dim)
        attn <- torch::torch_matmul(q, k$transpose(3, 4)) * scale
        attn <- torch::nnf_softmax(attn, dim = - 1)
        out <- torch::torch_matmul(attn, v)

        # Reshape back to (batch, q_len, embed_dim)
        out <- out$transpose(2, 3)$contiguous()$view(c(batch_size, q_len, self$embed_dim))

        # Output projection and residual connection
        h <- self$proj_out$forward(out)
        x1 + h
    }
)

# ============================================================================
# Perceiver Resampler (matches Python architecture)
# ============================================================================

#' Perceiver resampler for conditioning compression
#'
#' @param num_query_tokens Number of query tokens (default 32)
#' @param embed_dim Embedding dimension (default 1024)
#' @param num_heads Number of attention heads (default 4)
#' @return nn_module
perceiver_resampler <- torch::nn_module(
    "Perceiver",

    initialize = function (num_query_tokens = 32, embed_dim = 1024,
                           num_heads = 4)
    {
        self$embed_dim <- embed_dim
        self$num_heads <- num_heads

        # Learnable query tokens (pre_attention_query in Python)
        query_var <- sqrt(3.0) * sqrt(2.0 / (num_query_tokens + num_query_tokens))
        self$pre_attention_query <- torch::nn_parameter(
            torch::torch_empty(1, num_query_tokens, embed_dim)$uniform_(- query_var, query_var)
        )

        # Single attention block (reused for cross-attention and self-attention)
        self$attn <- attention_block(embed_dim, num_heads)
    },

    forward = function (x)
    {
        batch_size <- x$size(1)

        # Expand query to batch size
        query <- self$pre_attention_query$expand(c(batch_size, - 1, - 1))

        # Cross-attention: query attends to input
        pre_att <- self$attn$forward(query, x)

        # Self-attention: result attends to itself
        out <- self$attn$forward(pre_att, pre_att)

        out
    }
)

# ============================================================================
# T3 Conditioning Encoder
# ============================================================================

#' T3 conditioning encoder
#'
#' @param config T3 configuration
#' @return nn_module
t3_cond_enc <- torch::nn_module(
    "T3CondEnc",

    initialize = function (config = NULL)
    {
        if (is.null(config)) {
            config <- t3_config_english()
        }
        self$config <- config

        # Speaker embedding projection
        self$spkr_enc <- torch::nn_linear(config$speaker_embed_size, config$n_channels)

        # Emotion projection
        self$emotion_adv_fc <- NULL
        if (config$emotion_adv) {
            self$emotion_adv_fc <- torch::nn_linear(1, config$n_channels, bias = FALSE)
        }

        # Perceiver resampler
        self$perceiver <- NULL
        if (config$use_perceiver_resampler) {
            self$perceiver <- perceiver_resampler(32, config$n_channels, 4)
        }
    },

    forward = function (cond)
    {
        # Get device from speaker embedding
        device <- cond$speaker_emb$device
        batch_size <- cond$speaker_emb$size(1)

        # Speaker embedding projection: (B, 256) -> (B, 1, 1024)
        cond_spkr <- self$spkr_enc$forward(cond$speaker_emb$view(c(- 1, self$config$speaker_embed_size)))
        cond_spkr <- cond_spkr$unsqueeze(2)

        # Empty tensor for unused conditioning
        empty <- torch::torch_zeros(c(batch_size, 0, self$config$n_channels), device = device)

        # CLAP not implemented
        cond_clap <- empty

        # Conditional prompt speech embeddings
        cond_prompt_speech_emb <- cond$cond_prompt_speech_emb
        if (is.null(cond_prompt_speech_emb)) {
            cond_prompt_speech_emb <- empty
        } else if (!is.null(self$perceiver)) {
            cond_prompt_speech_emb <- self$perceiver$forward(cond_prompt_speech_emb)
        }

        # Emotion control
        cond_emotion_adv <- empty
        if (!is.null(self$emotion_adv_fc) && !is.null(cond$emotion_adv)) {
            emotion_val <- cond$emotion_adv
            if (!inherits(emotion_val, "torch_tensor")) {
                emotion_val <- torch::torch_tensor(emotion_val, device = device)
            }
            emotion_val <- emotion_val$view(c(- 1, 1, 1))
            cond_emotion_adv <- self$emotion_adv_fc$forward(emotion_val)
        }

        # Concatenate all conditioning
        torch::torch_cat(list(cond_spkr, cond_clap, cond_prompt_speech_emb, cond_emotion_adv), dim = 2)
    }
)

# ============================================================================
# T3 Model
# ============================================================================

#' T3 Token-to-Token TTS model
#'
#' @param config T3 configuration
#' @return nn_module
t3_model <- torch::nn_module(
    "T3Model",

    initialize = function (config = NULL)
    {
        if (is.null(config)) {
            config <- t3_config_english()
        }
        self$config <- config

        # Get Llama config
        llama_cfg <- llama_config_520m()
        self$llama_dim <- llama_cfg$hidden_size

        # Conditioning encoder
        self$cond_enc <- t3_cond_enc(config)

        # Token embeddings
        self$text_emb <- torch::nn_embedding(config$text_tokens_dict_size, self$llama_dim)
        self$speech_emb <- torch::nn_embedding(config$speech_tokens_dict_size, self$llama_dim)

        # Position embeddings
        max_text_seq_len <- config$max_text_tokens + 2
        max_speech_seq_len <- config$max_speech_tokens + 4
        self$text_pos_emb <- learned_position_embeddings(max_text_seq_len, self$llama_dim)
        self$speech_pos_emb <- learned_position_embeddings(max_speech_seq_len, self$llama_dim)

        # Output heads
        self$text_head <- torch::nn_linear(self$llama_dim, config$text_tokens_dict_size, bias = FALSE)
        self$speech_head <- torch::nn_linear(self$llama_dim, config$speech_tokens_dict_size, bias = FALSE)

        # Llama backbone
        self$tfmr <- llama_model(llama_cfg)
    },

    prepare_conditioning = function (cond)
    {
        # Embed speech tokens if provided but not yet embedded
        if (!is.null(cond$cond_prompt_speech_tokens) && is.null(cond$cond_prompt_speech_emb)) {
            tokens <- cond$cond_prompt_speech_tokens
            emb <- self$speech_emb$forward(tokens$add(1L)) + self$speech_pos_emb$forward(tokens)
            cond$cond_prompt_speech_emb <- emb
        }
        self$cond_enc$forward(cond)
    },

    prepare_input_embeds = function (cond, text_tokens, speech_tokens,
                                     cfg_weight = 0.0)
    {
        # Prepare conditioning embeddings
        cond_emb <- self$prepare_conditioning(cond) # (B, len_cond, dim)

        # Text embeddings with position
        text_emb <- self$text_emb$forward(text_tokens$add(1L)) # +1 for R indexing
        text_emb <- text_emb + self$text_pos_emb$forward(text_tokens)

        # Zero out text for CFG unconditional path
        if (cfg_weight > 0.0 && text_emb$size(1) > 1) {
            # Second batch element is unconditional
            text_emb[2,,] <- 0
        }

        # Speech embeddings with position
        speech_emb <- self$speech_emb$forward(speech_tokens$add(1L))
        speech_emb <- speech_emb + self$speech_pos_emb$forward(speech_tokens)

        len_cond <- cond_emb$size(2)

        # Expand conditioning if batch sizes don't match
        if (cond_emb$size(1) != text_emb$size(1)) {
            cond_emb <- cond_emb$expand(c(text_emb$size(1), - 1, - 1))
        }

        # Concatenate: cond + text + speech
        embeds <- torch::torch_cat(list(cond_emb, text_emb, speech_emb), dim = 2)

        list(embeds = embeds, len_cond = len_cond)
    },

    forward = function (cond, text_tokens, speech_tokens, cfg_weight = 0.0)
    {
        # Prepare embeddings
        prep <- self$prepare_input_embeds(cond, text_tokens, speech_tokens, cfg_weight)
        embeds <- prep$embeds
        len_cond <- prep$len_cond

        # Forward through Llama
        output <- self$tfmr$forward(inputs_embeds = embeds, use_cache = FALSE,
            output_hidden_states = TRUE)

        hidden_states <- output$last_hidden_state

        # Extract text and speech portions
        len_text <- text_tokens$size(2)
        len_speech <- speech_tokens$size(2)

        text_start <- len_cond + 1
        text_end <- len_cond + len_text
        speech_start <- len_cond + len_text + 1
        speech_end <- len_cond + len_text + len_speech

        text_latents <- hidden_states[, text_start:text_end,]
        speech_latents <- hidden_states[, speech_start:speech_end,]

        # Project to logits
        text_logits <- self$text_head$forward(text_latents)
        speech_logits <- self$speech_head$forward(speech_latents)

        list(
            text_logits = text_logits,
            speech_logits = speech_logits,
            text_latents = text_latents,
            speech_latents = speech_latents,
            hidden_states = hidden_states
        )
    }
)

# ============================================================================
# T3 Inference
# ============================================================================

#' Run T3 inference to generate speech tokens
#'
#' @param model T3 model
#' @param cond T3 conditioning
#' @param text_tokens Tokenized text (tensor)
#' @param max_new_tokens Maximum speech tokens to generate
#' @param temperature Sampling temperature
#' @param cfg_weight Classifier-free guidance weight
#' @param top_p Nucleus sampling threshold
#' @param min_p Minimum probability threshold
#' @param repetition_penalty Repetition penalty
#' @return Generated speech tokens
#' @export
t3_inference <- function (model, cond, text_tokens, max_new_tokens = 1000,
                          temperature = 0.8, cfg_weight = 0.5, top_p = 0.95,
                          min_p = 0.05, repetition_penalty = 1.2)
{
    config <- model$config
    device <- model$text_emb$weight$device

    # Ensure text_tokens is 2D
    if (text_tokens$dim() == 1) {
        text_tokens <- text_tokens$unsqueeze(1)
    }

    # Add start/stop text tokens
    sot <- config$start_text_token
    eot <- config$stop_text_token
    text_tokens <- torch::nnf_pad(text_tokens, c(1, 0), value = sot)
    text_tokens <- torch::nnf_pad(text_tokens, c(0, 1), value = eot)

    # Double batch for CFG
    if (cfg_weight > 0.0) {
        text_tokens <- torch::torch_cat(list(text_tokens, text_tokens), dim = 1)
    }

    # Initial speech token (BOS)
    bos_token <- torch::torch_tensor(matrix(config$start_speech_token, nrow = 1),
        device = device, dtype = torch::torch_long())

    # Double BOS token for CFG
    if (cfg_weight > 0.0) {
        bos_token <- torch::torch_cat(list(bos_token, bos_token), dim = 1)
    }

    # Prepare initial embeddings
    prep <- model$prepare_input_embeds(cond, text_tokens, bos_token, cfg_weight)
    embeds <- prep$embeds

    # Double BOS embedding for CFG
    if (cfg_weight > 0.0) {
        bos_emb <- model$speech_emb$forward(bos_token$add(1L)) + model$speech_pos_emb$get_fixed_embedding(0)
        bos_emb <- torch::torch_cat(list(bos_emb, bos_emb), dim = 1)
    }

    # Initial forward pass
    torch::with_no_grad({
            output <- model$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE)
            past_key_values <- output$past_key_values

            # Track generated tokens (only conditional path for CFG)
            generated_ids <- bos_token[1,, drop = FALSE]$clone()
            predicted <- list()

            # Generation loop
            for (i in seq_len(max_new_tokens)) {
                logits <- output$last_hidden_state[, - 1,]
                logits <- model$speech_head$forward(logits) # (B, vocab_size)

                # CFG combination
                if (cfg_weight > 0.0) {
                    cond_logits <- logits[1,]$unsqueeze(1)
                    uncond_logits <- logits[2,]$unsqueeze(1)
                    logits <- cond_logits + cfg_weight * (cond_logits - uncond_logits)
                } else {
                    logits <- logits[1,]$unsqueeze(1)
                }

                # Apply repetition penalty
                # Note: generated_ids contains 1-indexed values (from sorted_indices),
                # which is already correct for R tensor indexing (no +1 needed)
                if (repetition_penalty != 1.0) {
                    for (token_id in as.integer(generated_ids$cpu())) {
                        logits[1, token_id] <- logits[1, token_id] / repetition_penalty
                    }
                }

                # Temperature scaling
                if (temperature != 1.0) {
                    logits <- logits / temperature
                }

                # Min-p filtering
                probs <- torch::nnf_softmax(logits, dim = - 1)
                max_prob <- probs$max()
                min_threshold <- min_p * max_prob
                # Use -65504 instead of -Inf for float16 compatibility
                logits[probs < min_threshold] <- -65504.0

                # Recompute probs after min-p filtering
                probs_filtered <- torch::nnf_softmax(logits, dim = - 1)

                # Top-p (nucleus) sampling
                sorted_result <- torch::torch_sort(probs_filtered, descending = TRUE)
                sorted_probs <- sorted_result[[1]]
                sorted_indices <- sorted_result[[2]]
                cumsum_probs <- torch::torch_cumsum(sorted_probs, dim = - 1)

                # Remove tokens with cumulative probability above threshold
                sorted_mask <- cumsum_probs > top_p
                # Shift mask right to keep at least one token
                sorted_mask[, 1] <- FALSE
                sorted_probs[sorted_mask] <- 0

                # Re-normalize
                sorted_probs <- sorted_probs / sorted_probs$sum()

                # Sample
                next_token_idx <- torch::torch_multinomial(sorted_probs, num_samples = 1)
                next_token <- sorted_indices$gather(2, next_token_idx)

                predicted[[length(predicted) + 1]] <- next_token
                generated_ids <- torch::torch_cat(list(generated_ids, next_token), dim = 2)

                # Check for EOS
                # Note: sorted_indices returns R 1-indexed values, subtract 1 to get 0-indexed token ID
                token_id <- as.integer(next_token$cpu()) - 1L
                if (token_id == config$stop_speech_token) {
                    message("EOS detected at step ", i)
                    break
                }

                # Get embedding for next token
                # sorted_indices returns R 1-indexed values, which nn_embedding expects (no +1 needed)
                next_emb <- model$speech_emb$forward(next_token) + model$speech_pos_emb$get_fixed_embedding(i)

                # Double for CFG
                if (cfg_weight > 0.0) {
                    next_emb <- torch::torch_cat(list(next_emb, next_emb), dim = 1)
                }

                # Forward with KV cache
                output <- model$tfmr$forward(inputs_embeds = next_emb, past_key_values = past_key_values,
                    use_cache = TRUE)
                past_key_values <- output$past_key_values
            }
        })

    # Concatenate predicted tokens and convert to 0-indexed token IDs
    # (sorted_indices returns R 1-indexed positions)
    if (length(predicted) > 0) {
        tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)
        tokens <- tokens$sub(1L) # Convert to 0-indexed token IDs
        tokens
    } else {
        torch::torch_tensor(integer(0), device = device)
    }
}

# ============================================================================
# Weight Loading
# ============================================================================

#' Load T3 weights from safetensors
#'
#' @param model T3 model
#' @param state_dict Named list of tensors
#' @return Model with loaded weights
load_t3_weights <- function (model, state_dict)
{
    # Load Llama backbone weights (already wrapped in with_no_grad)
    llama_weights <- list()
    for (name in names(state_dict)) {
        if (startsWith(name, "tfmr.")) {
            new_name <- sub("^tfmr\\.", "", name)
            llama_weights[[new_name]] <- state_dict[[name]]
        }
    }
    load_llama_weights(model$tfmr, llama_weights, prefix = "")

    # Load remaining weights with no_grad
    torch::with_no_grad({
            # Load embedding weights
            if ("text_emb.weight" %in% names(state_dict)) {
                model$text_emb$weight$copy_(state_dict[["text_emb.weight"]])
            }
            if ("speech_emb.weight" %in% names(state_dict)) {
                model$speech_emb$weight$copy_(state_dict[["speech_emb.weight"]])
            }

            # Load position embedding weights
            if ("text_pos_emb.emb.weight" %in% names(state_dict)) {
                model$text_pos_emb$emb$weight$copy_(state_dict[["text_pos_emb.emb.weight"]])
            }
            if ("speech_pos_emb.emb.weight" %in% names(state_dict)) {
                model$speech_pos_emb$emb$weight$copy_(state_dict[["speech_pos_emb.emb.weight"]])
            }

            # Load output head weights
            if ("text_head.weight" %in% names(state_dict)) {
                model$text_head$weight$copy_(state_dict[["text_head.weight"]])
            }
            if ("speech_head.weight" %in% names(state_dict)) {
                model$speech_head$weight$copy_(state_dict[["speech_head.weight"]])
            }

            # Load conditioning encoder weights
            if ("cond_enc.spkr_enc.weight" %in% names(state_dict)) {
                model$cond_enc$spkr_enc$weight$copy_(state_dict[["cond_enc.spkr_enc.weight"]])
            }
            if ("cond_enc.spkr_enc.bias" %in% names(state_dict)) {
                model$cond_enc$spkr_enc$bias$copy_(state_dict[["cond_enc.spkr_enc.bias"]])
            }
            if ("cond_enc.emotion_adv_fc.weight" %in% names(state_dict)) {
                model$cond_enc$emotion_adv_fc$weight$copy_(state_dict[["cond_enc.emotion_adv_fc.weight"]])
            }

            # Load perceiver weights
            # Helper to copy if exists
            copy_if_exists <- function (r_param, key)
            {
                if (key %in% names(state_dict)) {
                    tryCatch({
                            r_param$copy_(state_dict[[key]])
                            return(TRUE)
                        }, error = function (e)
                        {
                            warning("Failed to copy ", key, ": ", e$message)
                            return(FALSE)
                        })
                }
                FALSE
            }

            # Perceiver query tokens (pre_attention_query in Python)
            copy_if_exists(model$cond_enc$perceiver$pre_attention_query, "cond_enc.perceiver.pre_attention_query")

            # Attention block (single attn module used for both cross and self-attention)
            copy_if_exists(model$cond_enc$perceiver$attn$norm$weight, "cond_enc.perceiver.attn.norm.weight")
            copy_if_exists(model$cond_enc$perceiver$attn$norm$bias, "cond_enc.perceiver.attn.norm.bias")

            copy_if_exists(model$cond_enc$perceiver$attn$to_q$weight, "cond_enc.perceiver.attn.to_q.weight")
            copy_if_exists(model$cond_enc$perceiver$attn$to_q$bias, "cond_enc.perceiver.attn.to_q.bias")
            copy_if_exists(model$cond_enc$perceiver$attn$to_k$weight, "cond_enc.perceiver.attn.to_k.weight")
            copy_if_exists(model$cond_enc$perceiver$attn$to_k$bias, "cond_enc.perceiver.attn.to_k.bias")
            copy_if_exists(model$cond_enc$perceiver$attn$to_v$weight, "cond_enc.perceiver.attn.to_v.weight")
            copy_if_exists(model$cond_enc$perceiver$attn$to_v$bias, "cond_enc.perceiver.attn.to_v.bias")
            copy_if_exists(model$cond_enc$perceiver$attn$proj_out$weight, "cond_enc.perceiver.attn.proj_out.weight")
            copy_if_exists(model$cond_enc$perceiver$attn$proj_out$bias, "cond_enc.perceiver.attn.proj_out.bias")
        })

    model
}

