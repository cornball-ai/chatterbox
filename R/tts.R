# chatterbox - High-level Text-to-Speech API
# Provides simple interface for TTS generation using the Chatterbox engine

# ============================================================================
# Chatterbox TTS Model
# ============================================================================

#' Create Chatterbox TTS model
#'
#' @param device Device to use ("cpu", "cuda", "mps", etc.)
#' @param turbo Use turbo model (GPT-2 backbone, MeanFlow decoder). Default FALSE.
#' @return Chatterbox TTS model object
#' @export
chatterbox <- function (device = "cpu", turbo = FALSE)
{
    structure(
        list(
            device = device,
            turbo = turbo,
            t3 = NULL,
            s3gen = NULL,
            voice_encoder = NULL,
            tokenizer = NULL,
            loaded = FALSE
        ),
        class = "chatterbox"
    )
}

#' Load Chatterbox model weights
#'
#' Load pretrained weights for all model components.
#' Requires prior download via \code{\link{download_chatterbox_models}}.
#'
#' @param model Chatterbox model object
#' @return Chatterbox model with loaded weights
#' @export
load_chatterbox <- function (model)
{
    if (!inherits(model, "chatterbox")) {
        stop("model must be a chatterbox object")
    }

    # Dispatch to turbo loader if turbo mode
    if (isTRUE(model$turbo)) {
        return(load_chatterbox_turbo(model))
    }

    device <- model$device
    message("Loading Chatterbox TTS model to ", device, "...")

    # Get model file paths (requires prior download)
    message("Loading model files...")
    paths <- get_model_paths()

    # Load tokenizer
    message("Loading text tokenizer...")
    model$tokenizer <- load_bpe_tokenizer(paths$tokenizer)

    # Load weights to CPU first, then move to device
    # This halves peak VRAM usage (avoids having weights in both dict and model)

    # Load voice encoder
    message("Loading voice encoder...")
    ve_weights <- read_safetensors(paths$ve, "cpu")
    model$voice_encoder <- voice_encoder()
    model$voice_encoder <- load_voice_encoder_weights(model$voice_encoder, ve_weights)
    rm(ve_weights); gc()
    model$voice_encoder$to(device = device)
    model$voice_encoder$eval()

    # Load T3 model
    message("Loading T3 text-to-speech model...")
    t3_weights <- read_safetensors(paths$t3_cfg, "cpu")
    model$t3 <- t3_model() # Creates T3Model instance
    model$t3 <- load_t3_weights(model$t3, t3_weights)
    rm(t3_weights); gc()
    model$t3$to(device = device)
    model$t3$eval()

    # Load S3Gen model
    message("Loading S3Gen speech decoder...")
    model$s3gen <- load_s3gen(paths$s3gen, device)

    model$loaded <- TRUE
    message("Chatterbox TTS model loaded successfully!")

    model
}

#' Load Chatterbox Turbo model weights
#'
#' Loads the turbo variant (GPT-2 backbone, MeanFlow decoder).
#' Requires prior download via \code{\link{download_chatterbox_turbo_models}}.
#'
#' @param model Chatterbox model object (with turbo=TRUE)
#' @return Chatterbox model with loaded weights
#' @export
load_chatterbox_turbo <- function (model)
{
    device <- model$device
    message("Loading Chatterbox Turbo model to ", device, "...")

    # Get turbo model file paths
    message("Loading turbo model files...")
    paths <- get_turbo_model_paths()

    # Load GPT-2 tokenizer
    message("Loading GPT-2 tokenizer...")
    model$tokenizer <- load_gpt2_tokenizer(
        paths$vocab,
        paths$merges,
        paths$added_tokens
    )

    # Load voice encoder (same as standard)
    message("Loading voice encoder...")
    ve_weights <- read_safetensors(paths$ve, "cpu")
    model$voice_encoder <- voice_encoder()
    model$voice_encoder <- load_voice_encoder_weights(model$voice_encoder, ve_weights)
    rm(ve_weights); gc()
    model$voice_encoder$to(device = device)
    model$voice_encoder$eval()

    # Load T3 turbo model (GPT-2 backbone)
    message("Loading T3 turbo model (GPT-2 backbone)...")
    t3_weights <- read_safetensors(paths$t3_turbo_v1, "cpu")
    model$t3 <- t3_model_turbo()
    model$t3 <- load_t3_turbo_weights(model$t3, t3_weights)
    rm(t3_weights); gc()
    model$t3$to(device = device)
    model$t3$eval()

    # Load S3Gen with MeanFlow decoder
    message("Loading S3Gen MeanFlow decoder...")
    model$s3gen <- load_s3gen(paths$s3gen_meanflow, device, meanflow = TRUE)

    model$loaded <- TRUE
    message("Chatterbox Turbo model loaded successfully!")

    model
}

#' Check if model is loaded
#'
#' @param model Chatterbox model
#' @return TRUE if model is loaded
is_loaded <- function (model)
{
    model$loaded
}

# ============================================================================
# Voice Embedding
# ============================================================================

#' Create voice embedding from reference audio
#'
#' @param model Chatterbox model
#' @param audio Reference audio (file path, numeric vector, or torch tensor)
#' @param sample_rate Sample rate of audio (if not a file)
#' @param autocast Ignored (kept for API compatibility)
#' @return Voice embedding that can be used for synthesis
#' @export
create_voice_embedding <- function (model, audio, sample_rate = NULL, autocast = NULL)
{
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }

    # Handle audio input
    if (is.character(audio)) {
        # Read audio file
        audio_data <- read_audio(audio)
        samples <- audio_data$samples
        sample_rate <- audio_data$sr
    } else if (is.numeric(audio)) {
        if (is.null(sample_rate)) {
            stop("sample_rate must be provided for numeric audio input")
        }
        samples <- audio
    } else if (inherits(audio, "torch_tensor")) {
        if (is.null(sample_rate)) {
            stop("sample_rate must be provided for tensor audio input")
        }
        samples <- as.numeric(audio$cpu())
    } else {
        stop("audio must be a file path, numeric vector, or torch tensor")
    }

    device <- model$device

    # Resample to 16kHz for voice encoder and tokenizer
    if (sample_rate != 16000) {
        samples_16k <- resample_audio(samples, sample_rate, 16000)
    } else {
        samples_16k <- samples
    }

    # Convert to tensor
    audio_tensor <- torch::torch_tensor(samples, dtype = torch::torch_float32())$unsqueeze(1)$to(device = device)
    audio_16k <- torch::torch_tensor(samples_16k, dtype = torch::torch_float32())$unsqueeze(1)$to(device = device)

    # Get voice encoder embedding using compute_speaker_embedding
    # (handles mel spectrogram computation internally)
    # Note: voice embedding runs in float32 for numerical stability
    torch::with_no_grad({
        ve_embedding <- compute_speaker_embedding(model$voice_encoder, audio_16k, 16000)
    })

    # Get conditioning prompt speech tokens from S3 tokenizer
    # Standard: 150 tokens, Turbo: 375 tokens
    cond_prompt_len <- model$t3$config$speech_cond_prompt_len
    torch::with_no_grad({
        tok_result <- model$s3gen$tokenizer$forward(audio_16k, max_len = cond_prompt_len)
        cond_prompt_tokens <- tok_result$tokens$to(device = device)
    })

    # Create reference dict for S3Gen
    torch::with_no_grad({
        ref_dict <- model$s3gen$embed_ref(audio_tensor$squeeze(1), sample_rate, device)
    })

    # Return voice embedding object
    structure(
        list(
            ve_embedding = ve_embedding,
            cond_prompt_speech_tokens = cond_prompt_tokens,
            ref_dict = ref_dict,
            sample_rate = sample_rate
        ),
        class = "voice_embedding"
    )
}

# ============================================================================
# Text-to-Speech Synthesis
# ============================================================================

#' Generate speech from text
#'
#' @param model Chatterbox model
#' @param text Text to synthesize
#' @param voice Voice embedding from create_voice_embedding() or path to reference audio
#' @param exaggeration Emotion/expression exaggeration level (0-1, default 0.5)
#' @param cfg_weight Classifier-free guidance weight (higher = more adherence to text, default 0.5)
#' @param temperature Sampling temperature (default 0.8)
#' @param top_p Top-p (nucleus) sampling threshold (default 0.9)
#' @param autocast Use mixed precision (float16) on CUDA for faster inference (default TRUE on CUDA)
#' @return List with audio (numeric vector) and sample_rate
#' @export
generate <- function (model, text, voice, exaggeration = 0.5, cfg_weight = 0.5,
                      temperature = 0.8, top_p = 0.9, autocast = NULL,
                      traced = FALSE, backend = c("r", "cpp"),
                      top_k = 1000L, repetition_penalty = 1.2)
{
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }

    device <- model$device
    is_turbo <- isTRUE(model$turbo)
    # Default: autocast on CUDA, off on CPU
    use_autocast <- if (is.null(autocast)) grepl("^cuda", device) else autocast

    # Handle voice input
    if (is.character(voice)) {
        voice <- create_voice_embedding(model, voice, autocast = use_autocast)
    } else if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object or path to reference audio")
    }

    # Tokenize text
    if (is_turbo) {
        # GPT-2 tokenizer
        text <- punc_norm(text)
        text_ids <- tokenize_text_gpt2(model$tokenizer, text)
        text_tokens <- torch::torch_tensor(text_ids, dtype = torch::torch_long())$unsqueeze(1L)$to(device = device)
    } else {
        text_tokens <- tokenize_text(model$tokenizer, text)
        text_tokens <- torch::torch_tensor(text_tokens, dtype = torch::torch_long())$unsqueeze(1L)$to(device = device)
    }

    # Create T3 conditioning
    cond <- t3_cond(
        speaker_emb = voice$ve_embedding,
        cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
        emotion_adv = if (is_turbo) NULL else exaggeration
    )

    # Generate speech tokens with T3
    message("Generating speech tokens...")

    if (is_turbo) {
        # Turbo inference: no CFG, no min_p, uses top_k
        if (use_autocast) {
            torch::with_autocast(device_type = "cuda", {
                torch::with_no_grad({
                    speech_tokens <- t3_inference_turbo(
                        model = model$t3,
                        cond = cond,
                        text_tokens = text_tokens,
                        temperature = temperature,
                        top_k = top_k,
                        top_p = top_p,
                        repetition_penalty = repetition_penalty
                    )
                })
            })
        } else {
            torch::with_no_grad({
                speech_tokens <- t3_inference_turbo(
                    model = model$t3,
                    cond = cond,
                    text_tokens = text_tokens,
                    temperature = temperature,
                    top_k = top_k,
                    top_p = top_p,
                    repetition_penalty = repetition_penalty
                )
            })
        }
    } else {
        # Standard inference with CFG
        backend <- match.arg(backend)
        if (backend == "cpp") {
            inference_fn <- t3_inference_cpp
        } else if (traced) {
            inference_fn <- t3_inference_traced
        } else {
            inference_fn <- t3_inference
        }

        if (use_autocast) {
            torch::with_autocast(device_type = "cuda", {
                torch::with_no_grad({
                    speech_tokens <- inference_fn(
                        model = model$t3,
                        cond = cond,
                        text_tokens = text_tokens,
                        cfg_weight = cfg_weight,
                        temperature = temperature,
                        top_p = top_p
                    )
                })
            })
        } else {
            torch::with_no_grad({
                speech_tokens <- inference_fn(
                    model = model$t3,
                    cond = cond,
                    text_tokens = text_tokens,
                    cfg_weight = cfg_weight,
                    temperature = temperature,
                    top_p = top_p
                )
            })
        }
    }

    # Drop invalid tokens
    speech_tokens <- drop_invalid_tokens(speech_tokens)

    if (length(speech_tokens) == 0) {
        warning("No valid speech tokens generated")
        return(list(audio = numeric(0), sample_rate = S3GEN_SR))
    }

    # Convert to integer vector and add silence for turbo
    token_vals <- as.integer(speech_tokens)
    if (is_turbo) {
        # Append 3x silence tokens
        token_vals <- c(token_vals, rep(S3GEN_SIL, 3L))
    }

    speech_tokens <- torch::torch_tensor(
        token_vals,
        dtype = torch::torch_long()
    )$unsqueeze(1L)$to(device = device)

    # Generate waveform with S3Gen
    message("Synthesizing waveform...")
    n_cfm_steps <- if (is_turbo) 2L else NULL

    if (use_autocast) {
        torch::with_autocast(device_type = "cuda", {
            torch::with_no_grad({
                result <- model$s3gen$inference(
                    speech_tokens = speech_tokens,
                    ref_dict = voice$ref_dict,
                    finalize = TRUE,
                    traced = traced,
                    n_cfm_timesteps = n_cfm_steps
                )
                audio <- result[[1]]
            })
        })
    } else {
        torch::with_no_grad({
            result <- model$s3gen$inference(
                speech_tokens = speech_tokens,
                ref_dict = voice$ref_dict,
                finalize = TRUE,
                traced = traced,
                n_cfm_timesteps = n_cfm_steps
            )
            audio <- result[[1]]
        })
    }

    # Convert to numeric
    audio_samples <- as.numeric(audio$squeeze()$cpu())

    message("Done! Generated ", round(length(audio_samples) / S3GEN_SR, 2), " seconds of audio.")

    list(
        audio = audio_samples,
        sample_rate = S3GEN_SR
    )
}

#' Generate speech and save to file
#'
#' @param model Chatterbox model
#' @param text Text to synthesize
#' @param voice Voice embedding or path to reference audio
#' @param output_path Output file path (WAV format)
#' @param ... Additional arguments passed to generate()
#' @return Invisibly returns the output path
#' @export
tts_to_file <- function (model, text, voice, output_path, ...)
{
    result <- generate(model, text, voice, ...)
    write_audio(result$audio, result$sample_rate, output_path)
    invisible(output_path)
}

# ============================================================================
# Streaming TTS (for longer texts)
# ============================================================================

#' Generate speech in chunks (for long texts)
#'
#' @param model Chatterbox model
#' @param text Text to synthesize
#' @param voice Voice embedding
#' @param chunk_size Maximum tokens per chunk
#' @param ... Additional arguments passed to generate()
#' @return List with audio and sample_rate
#' @export
tts_chunked <- function (model, text, voice, chunk_size = 200, ...)
{
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }

    # Split text into sentences
    sentences <- strsplit(text, "(?<=[.!?])\\s+", perl = TRUE) [[1]]

    all_audio <- numeric(0)

    for (i in seq_along(sentences)) {
        sentence <- sentences[i]
        message(sprintf("Processing chunk %d/%d: %s...",
                i, length(sentences),
                substr(sentence, 1, 50)))

        result <- generate(model, sentence, voice, ...)
        all_audio <- c(all_audio, result$audio)
    }

    list(
        audio = all_audio,
        sample_rate = S3GEN_SR
    )
}

# ============================================================================
# Print Methods
# ============================================================================

#' Print method for chatterbox
#'
#' @param x Chatterbox model
#' @param ... Ignored
#' @export
print.chatterbox <- function (x, ...)
{
    variant <- if (isTRUE(x$turbo)) "Turbo" else "Standard"
    cat("Chatterbox TTS Model (", variant, ")\n", sep = "")
    cat("  Device:", x$device, "\n")
    cat("  Loaded:", x$loaded, "\n")
    if (x$loaded) {
        cat("  Components:\n")
        if (isTRUE(x$turbo)) {
            cat("    - T3 Turbo (GPT-2 backbone)\n")
            cat("    - S3Gen MeanFlow (2-step decoder)\n")
        } else {
            cat("    - T3 (Llama backbone)\n")
            cat("    - S3Gen (10-step CFM decoder)\n")
        }
        cat("    - Voice Encoder\n")
        cat("    - Text Tokenizer\n")
    }
    invisible(x)
}

#' Print method for voice_embedding
#'
#' @param x Voice embedding
#' @param ... Ignored
#' @export
print.voice_embedding <- function (x, ...)
{
    cat("Voice Embedding\n")
    cat("  Speaker embedding shape:", paste(dim(x$ve_embedding), collapse = " x "), "\n")
    cat("  Reference sample rate:", x$sample_rate, "Hz\n")
    invisible(x)
}

# ============================================================================
# Convenience Functions
# ============================================================================

#' Quick TTS - one-line text-to-speech
#'
#' Loads model if needed and generates speech. Convenient for quick tests.
#'
#' @param text Text to synthesize
#' @param reference_audio Path to reference audio file
#' @param output_path Optional output file path. If NULL, returns audio data.
#' @param device Device to use
#' @param autocast Use mixed precision (float16) on CUDA (default TRUE on CUDA)
#' @return If output_path is NULL, returns list with audio and sample_rate.
#'         Otherwise writes to file and returns path invisibly.
#' @export
quick_tts <- function (text, reference_audio, output_path = NULL,
                       device = "cpu", autocast = NULL, turbo = FALSE)
{
    # Create and load model (caches after first load)
    model <- chatterbox(device, turbo = turbo)
    model <- load_chatterbox(model)

    # Generate
    if (is.null(output_path)) {
        generate(model, text, reference_audio, autocast = autocast)
    } else {
        tts_to_file(model, text, reference_audio, output_path, autocast = autocast)
    }
}

