#!/usr/bin/env Rscript
# Debug T3 inference step-by-step, comparing with Python

rhydrogen::load_all()

# Load Python debug outputs
debug_path <- "/home/troy/chatterbox/outputs/t3_debug.safetensors"
message("Loading Python debug from: ", debug_path)
py_debug <- read_safetensors(debug_path, device = "cpu")

cat("\n=== Python Debug Values ===\n")
cat("Raw logits shape:", paste(dim(py_debug$raw_logits_step0), collapse = "x"), "\n")
cat("Raw logits mean:", mean(as.numeric(py_debug$raw_logits_step0)), "\n")
cat("Raw logits range: [", min(as.numeric(py_debug$raw_logits_step0)), ", ",
    max(as.numeric(py_debug$raw_logits_step0)), "]\n", sep = "")

cat("\nPython sampled tokens:", paste(as.integer(py_debug$sampled_tokens), collapse = ", "), "\n")

# Load R model
cat("\n=== Loading R Model ===\n")
device <- "cuda"
model <- chatterbox(device)
model <- load_chatterbox(model)

# Use same reference audio
ref_audio <- "/home/troy/cornball_media/LilCasey/Casey_voice_samples/ShortCasey.wav"
text <- "Hello, this is a test of the text to speech system."

# Tokenize text
text_tokens <- tokenize_text(model$tokenizer, punc_norm(text))
text_tokens_tensor <- torch::torch_tensor(text_tokens, dtype = torch::torch_long())$unsqueeze(1)$to(device = device)

# Create conditioning - include cond_prompt_speech_tokens
voice <- create_voice_embedding(model, ref_audio)
cond <- t3_cond(
    speaker_emb = voice$ve_embedding,
    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
    emotion_adv = 0.5
)

cat("Cond prompt tokens shape:", paste(dim(cond$cond_prompt_speech_tokens), collapse = "x"), "\n")

# Get config
config <- model$t3$config
sot <- config$start_text_token
eot <- config$stop_text_token

# Add start/stop text tokens
text_tokens_padded <- torch::nnf_pad(text_tokens_tensor, c(1, 0), value = sot)
text_tokens_padded <- torch::nnf_pad(text_tokens_padded, c(0, 1), value = eot)

cat("\n=== R Text Tokens ===\n")
cat("Text tokens with SOT/EOT:", paste(as.integer(text_tokens_padded$cpu()), collapse = ", "), "\n")

# Double batch for CFG
cfg_weight <- 0.5
text_tokens_cfg <- torch::torch_cat(list(text_tokens_padded, text_tokens_padded), dim = 1)
cat("Text tokens CFG shape:", paste(dim(text_tokens_cfg), collapse = "x"), "\n")

# Set seed
set.seed(42)
torch::torch_manual_seed(42L)

# Initial BOS token
bos_token <- torch::torch_tensor(matrix(config$start_speech_token, nrow = 1),
    device = device, dtype = torch::torch_long())
bos_token_cfg <- torch::torch_cat(list(bos_token, bos_token), dim = 1)

cat("\n=== Prepare Input Embeddings ===\n")
# Prepare initial embeddings
prep <- model$t3$prepare_input_embeds(cond, text_tokens_cfg, bos_token_cfg, cfg_weight)
embeds <- prep$embeds
len_cond <- prep$len_cond
cat("Embeddings shape:", paste(dim(embeds), collapse = "x"), "\n")
cat("Conditioning length:", len_cond, "\n")

# Compare with Python
py_embeds <- py_debug$initial_embeds$to(device = device)
emb_diff <- (embeds - py_embeds)$abs()
cat("Embedding max diff from Python:", emb_diff$max()$item(), "\n")
cat("Embedding mean diff from Python:", emb_diff$mean()$item(), "\n")

cat("\n=== Initial Forward Pass ===\n")
# Forward pass
torch::with_no_grad({
        output <- model$t3$tfmr$forward(inputs_embeds = embeds, use_cache = TRUE,
            output_hidden_states = TRUE)
        past_key_values <- output$past_key_values

        # Get raw logits
        hidden_states <- output$last_hidden_state
        cat("Hidden states shape:", paste(dim(hidden_states), collapse = "x"), "\n")

        # Get last position
        last_hidden <- hidden_states[, hidden_states$size(2),]
        cat("Last hidden shape:", paste(dim(last_hidden), collapse = "x"), "\n")
        cat("Last hidden mean:", last_hidden$mean()$item(), "\n")
        cat("Last hidden std:", last_hidden$std()$item(), "\n")

        # Apply speech head
        logits_raw <- model$t3$speech_head$forward(last_hidden)
        cat("\nRaw logits shape:", paste(dim(logits_raw), collapse = "x"), "\n")
        cat("Raw logits mean:", logits_raw$mean()$item(), "\n")
        cat("Raw logits std:", logits_raw$std()$item(), "\n")
        cat("Raw logits range: [", logits_raw$min()$item(), ", ", logits_raw$max()$item(), "]\n", sep = "")

        # Compare with Python
        py_raw_logits <- py_debug$raw_logits_step0$to(device = device)
        raw_diff <- (logits_raw - py_raw_logits)$abs()
        cat("\nLogits diff from Python - max:", raw_diff$max()$item(), ", mean:", raw_diff$mean()$item(), "\n")

        # CFG combination
        cat("\n=== CFG Combination ===\n")
        cond_logits <- logits_raw[1,]$unsqueeze(1) # (1, vocab)
        uncond_logits <- logits_raw[2,]$unsqueeze(1) # (1, vocab)
        cat("Cond logits shape:", paste(dim(cond_logits), collapse = "x"), "\n")
        cat("Uncond logits shape:", paste(dim(uncond_logits), collapse = "x"), "\n")

        logits_cfg <- cond_logits + cfg_weight * (cond_logits - uncond_logits)
        cat("After CFG - shape:", paste(dim(logits_cfg), collapse = "x"), "\n")
        cat("After CFG - mean:", logits_cfg$mean()$item(), "\n")
        cat("After CFG - range: [", logits_cfg$min()$item(), ", ", logits_cfg$max()$item(), "]\n", sep = "")

        # Compare with Python
        py_cfg <- py_debug$logits_after_cfg$to(device = device)
        cfg_diff <- (logits_cfg - py_cfg)$abs()
        cat("CFG diff from Python - max:", cfg_diff$max()$item(), ", mean:", cfg_diff$mean()$item(), "\n")

        # Apply repetition penalty
        cat("\n=== Repetition Penalty ===\n")
        generated_ids <- bos_token$clone()
        repetition_penalty <- 1.2
        logits_rep <- logits_cfg$clone()
        bos_id <- as.integer(generated_ids$cpu())[1]
        cat("BOS token id:", bos_id, "\n")
        # Apply penalty to BOS token position (0-indexed token, 1-indexed R)
        old_val <- logits_rep[1, bos_id + 1]$item()
        logits_rep[1, bos_id + 1] <- logits_rep[1, bos_id + 1] / repetition_penalty
        new_val <- logits_rep[1, bos_id + 1]$item()
        cat("Logit at BOS position changed from", old_val, "to", new_val, "\n")

        # Apply temperature
        cat("\n=== Temperature ===\n")
        temperature <- 0.8
        logits_temp <- logits_rep / temperature
        cat("After temperature - range: [", logits_temp$min()$item(), ", ", logits_temp$max()$item(), "]\n", sep = "")

        # Compare with Python
        py_temp <- py_debug$logits_after_temp$to(device = device)
        temp_diff <- (logits_temp - py_temp)$abs()
        cat("Temperature diff from Python - max:", temp_diff$max()$item(), ", mean:", temp_diff$mean()$item(), "\n")

        # Apply min_p filtering
        cat("\n=== Min-P Filtering ===\n")
        min_p <- 0.05
        probs_for_minp <- torch::nnf_softmax(logits_temp, dim = - 1)
        max_prob <- probs_for_minp$max()$item()
        min_threshold <- min_p * max_prob
        cat("Max prob:", max_prob, "\n")
        cat("Min-p threshold:", min_threshold, "\n")

        # Set logits below threshold to -Inf
        logits_minp <- logits_temp$clone()
        mask_minp <- probs_for_minp < min_threshold
        logits_minp[mask_minp] <- - Inf
        non_inf_count <- (logits_minp > - Inf)$sum()$item()
        cat("Non-inf count after min_p:", non_inf_count, "\n")

        # Apply top_p filtering
        cat("\n=== Top-P Filtering ===\n")
        top_p <- 0.9
        probs_for_topp <- torch::nnf_softmax(logits_minp, dim = - 1) # Recompute after min_p
        sorted_result <- torch::torch_sort(probs_for_topp, descending = TRUE)
        sorted_probs <- sorted_result[[1]]
        sorted_indices <- sorted_result[[2]]
        cumsum_probs <- torch::torch_cumsum(sorted_probs, dim = - 1)

        # Create mask for tokens beyond top_p
        sorted_mask <- cumsum_probs > top_p
        # Keep at least one token
        sorted_mask[, 1] <- FALSE

        # Apply mask
        sorted_probs_filtered <- sorted_probs$clone()
        sorted_probs_filtered[sorted_mask] <- 0

        # Normalize
        sorted_probs_norm <- sorted_probs_filtered / sorted_probs_filtered$sum()

        non_zero_count <- (sorted_probs_norm > 0)$sum()$item()
        cat("Non-zero prob count after top_p:", non_zero_count, "\n")
        cat("Max prob after filtering:", sorted_probs_norm$max()$item(), "\n")

        # Get top 10 tokens
        cat("\nTop 10 tokens by probability:\n")
        top_probs <- sorted_probs_norm[1, 1:10]
        top_indices <- sorted_indices[1, 1:10]
        for (i in 1:10) {
            cat("  Token", as.integer(top_indices[i]$cpu()), ":", as.numeric(top_probs[i]$cpu()), "\n")
        }

        # Compare with Python probs
        cat("\n=== Compare with Python Probs ===\n")
        py_probs <- py_debug$probs_step0$to(device = device)
        cat("Python max prob:", py_probs$max()$item(), "\n")
        cat("Python non-zero count:", (py_probs > 0)$sum()$item(), "\n")

        # Sample
        cat("\n=== Sampling ===\n")
        next_token_idx <- torch::torch_multinomial(sorted_probs_norm, num_samples = 1)
        next_token <- sorted_indices$gather(2, next_token_idx)
        sampled_id <- as.integer(next_token$cpu())
        cat("Sampled token index in sorted:", as.integer(next_token_idx$cpu()), "\n")
        cat("Sampled token id:", sampled_id, "\n")

        cat("\n=== Summary ===\n")
        cat("Python sampled tokens:", paste(as.integer(py_debug$sampled_tokens), collapse = ", "), "\n")
        cat("R first sampled token:", sampled_id, "\n")

        # Check if embedding diff is the root cause
        if (emb_diff$max()$item() > 0.01) {
            cat("\nWARNING: Embedding difference is significant\n")
            cat("This could cause cascading differences in generation\n")
        }

        if (raw_diff$max()$item() > 0.1) {
            cat("\nWARNING: Raw logits differ significantly\n")
            cat("Check Llama backbone implementation\n")
        }

        if (cfg_diff$max()$item() > 0.1) {
            cat("\nWARNING: CFG logits differ significantly\n")
            cat("Check CFG implementation\n")
        }
    })

