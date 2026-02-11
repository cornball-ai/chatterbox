# chatterbox - High-level Text-to-Speech API
# Provides simple interface for TTS generation using the Chatterbox engine

# ============================================================================
# Chatterbox TTS Model
# ============================================================================

#' Create Chatterbox TTS model
#'
#' @param device Device to use ("cpu", "cuda", "mps", etc.)
#' @return Chatterbox TTS model object
#' @export
chatterbox <- function (device = "cpu")
{
    structure(
        list(
            device = device,
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
#' @param compiled If TRUE, compile eligible sub-modules using
#'   \code{Rtorch::compile()} for fused kernel execution.
#' @return Chatterbox model with loaded weights
#' @export
load_chatterbox <- function (model, compiled = FALSE)
{
    if (!inherits(model, "chatterbox")) {
        stop("model must be a chatterbox object")
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

    if (compiled) {
        message("Compiling model sub-modules...")
        model <- compile_chatterbox(model)
    }

    message("Chatterbox TTS model loaded successfully!")

    model
}

# ============================================================================
# Compilation
# ============================================================================

#' Compile a sub-module's forward method
#'
#' Traces the module with Rtorch::compile() and patches its forward
#' method to use the compiled path. Returns the module unchanged if
#' graph breaks are detected.
#'
#' @param module An nn_module
#' @param label Human-readable name for messages
#' @param ... Named example tensors matching forward() signature
#' @return The module (forward patched in place if compilation succeeded)
compile_module_forward <- function (module, label = NULL, ...)
{
    compiled <- tryCatch(
        Rtorch::compile(module, ...),
        error = function (e) {
            if (!is.null(label)) message("  skip: ", label, " (", conditionMessage(e), ")")
            NULL
        }
    )
    if (is.null(compiled)) return(module)

    if (length(compiled$graph_breaks) > 0) {
        reasons <- vapply(compiled$graph_breaks, `[[`, character(1), "reason")
        if (!is.null(label)) {
            message("  skip: ", label, " (graph breaks: ", paste(reasons, collapse = "; "), ")")
        }
        return(module)
    }

    # Patch forward to use compiled fn
    env <- attr(module, ".module_env")
    env$forward <- function (...) compiled$fn(...)
    if (!is.null(label)) message("  compiled: ", label)
    module
}

#' Compile chatterbox model sub-modules
#'
#' Walks the loaded model and compiles eligible sub-modules using
#' Rtorch's torchlang compiler. Compiled modules fuse elementwise
#' operation chains into single kernel calls.
#'
#' @param model A loaded chatterbox model
#' @return The model with compiled sub-modules
#' @export
compile_chatterbox <- function (model)
{
    device <- model$device
    llama_dim <- 1024L
    cfm_dim <- 256L

    # Example tensors for tracing
    ex_llama <- Rtorch::torch_randn(c(1L, 1L, llama_dim), device = device)
    ex_cfm <- Rtorch::torch_randn(c(1L, 1L, cfm_dim), device = device)
    ex_mish <- Rtorch::torch_randn(c(1L, cfm_dim, 10L), device = device)

    # -- T3 Llama MLP layers (30 instances) --
    message("  T3 Llama MLP layers...")
    layers <- model$t3$tfmr$layers
    for (i in seq_along(layers)) {
        layer <- layers[[i]]
        compile_module_forward(layer$mlp, label = sprintf("llama_mlp[%d]", i), x = ex_llama)
    }

    # -- CFM estimator sub-modules --
    est <- model$s3gen$flow$decoder$estimator
    if (!is.null(est)) {
        # Mish activation inside causal blocks
        # These are nested: resnet -> block1/block2 -> mish
        message("  CFM mish activations...")
        compile_mish <- function (block, prefix) {
            if (!is.null(block$block1$mish)) {
                compile_module_forward(block$block1$mish, label = paste0(prefix, ".block1.mish"),
                                       x = ex_mish)
            }
            if (!is.null(block$block2$mish)) {
                compile_module_forward(block$block2$mish, label = paste0(prefix, ".block2.mish"),
                                       x = ex_mish)
            }
        }

        if (!is.null(est$down_resnet)) compile_mish(est$down_resnet, "down_resnet")
        for (i in seq_along(est$mid_resnets)) {
            compile_mish(est$mid_resnets[[i]], sprintf("mid_resnets[%d]", i))
        }
        if (!is.null(est$up_resnet)) compile_mish(est$up_resnet, "up_resnet")

        # GELU projections inside transformer feed-forward nets
        message("  CFM GELU projections...")
        compile_gelu <- function (tfm_block, prefix) {
            gelu_mod <- tfm_block$ff$net[[1]]
            if (!is.null(gelu_mod)) {
                compile_module_forward(gelu_mod, label = paste0(prefix, ".ff.gelu"),
                                       x = ex_cfm)
            }
        }

        for (i in seq_along(est$down_transformers)) {
            compile_gelu(est$down_transformers[[i]], sprintf("down_tfm[%d]", i))
        }
        for (i in seq_along(est$mid_transformers)) {
            for (j in seq_along(est$mid_transformers[[i]])) {
                compile_gelu(est$mid_transformers[[i]][[j]], sprintf("mid_tfm[%d][%d]", i, j))
            }
        }
        for (i in seq_along(est$up_transformers)) {
            compile_gelu(est$up_transformers[[i]], sprintf("up_tfm[%d]", i))
        }
    }

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
    audio_tensor <- Rtorch::torch_tensor(samples, dtype = Rtorch::torch_float32)$unsqueeze(1)$to(device = device)
    audio_16k <- Rtorch::torch_tensor(samples_16k, dtype = Rtorch::torch_float32)$unsqueeze(1)$to(device = device)

    # Get voice encoder embedding using compute_speaker_embedding
    # (handles mel spectrogram computation internally)
    # Note: voice embedding runs in float32 for numerical stability
    Rtorch::with_no_grad({
        ve_embedding <- compute_speaker_embedding(model$voice_encoder, audio_16k, 16000)
    })

    # Get conditioning prompt speech tokens from S3 tokenizer
    # Python uses speech_cond_prompt_len = 150 tokens max
    cond_prompt_len <- model$t3$config$speech_cond_prompt_len # 150
    Rtorch::with_no_grad({
        tok_result <- model$s3gen$tokenizer$forward(audio_16k, max_len = cond_prompt_len)
        cond_prompt_tokens <- tok_result$tokens$to(device = device)
    })

    # Create reference dict for S3Gen
    Rtorch::with_no_grad({
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
                      traced = FALSE, backend = c("r", "cpp"))
{
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }

    device <- model$device
    # Default: autocast on CUDA, off on CPU
    use_autocast <- if (is.null(autocast)) grepl("^cuda", device) else autocast

    # Handle voice input
    if (is.character(voice)) {
        # It's a file path
        voice <- create_voice_embedding(model, voice, autocast = use_autocast)
    } else if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object or path to reference audio")
    }

    # Tokenize text
    text_tokens <- tokenize_text(model$tokenizer, text)
    text_tokens <- Rtorch::torch_tensor(text_tokens, dtype = Rtorch::torch_long)$unsqueeze(1)$to(device = device)

    # Create T3 conditioning with cond_prompt_speech_tokens
    cond <- t3_cond(
        speaker_emb = voice$ve_embedding,
        cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
        emotion_adv = exaggeration
    )

    # Generate speech tokens with T3
    message("Generating speech tokens...")

    # Select inference function
    backend <- match.arg(backend)
    if (backend == "cpp") {
        inference_fn <- t3_inference_cpp
    } else if (traced) {
        inference_fn <- t3_inference_traced
    } else {
        inference_fn <- t3_inference
    }

    if (use_autocast) {
        Rtorch::with_autocast(device_type = "cuda", code = {
            Rtorch::with_no_grad({
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
        Rtorch::with_no_grad({
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

    # Drop any invalid tokens
    speech_tokens <- drop_invalid_tokens(speech_tokens)

    if (length(speech_tokens) == 0) {
        warning("No valid speech tokens generated")
        return(list(audio = numeric(0), sample_rate = S3GEN_SR))
    }

    # Convert to tensor
    speech_tokens <- Rtorch::torch_tensor(
        as.integer(speech_tokens),
        dtype = Rtorch::torch_long
    )$unsqueeze(1)$to(device = device)

    # Offload T3 and voice encoder to CPU to free VRAM for S3Gen.
    # Also drop conditioning tensors no longer needed.
    if (grepl("^cuda", device)) {
        model$t3$to(device = "cpu")
        model$voice_encoder$to(device = "cpu")
        rm(cond, text_tokens)
        gc(); gc()
        Rtorch::cuda_empty_cache()
    }

    # Generate waveform with S3Gen
    message("Synthesizing waveform...")
    if (use_autocast) {
        Rtorch::with_autocast(device_type = "cuda", code = {
            Rtorch::with_no_grad({
                result <- model$s3gen$inference(
                    speech_tokens = speech_tokens,
                    ref_dict = voice$ref_dict,
                    finalize = TRUE,
                    traced = traced
                )
                audio <- result[[1]]
            })
        })
    } else {
        Rtorch::with_no_grad({
            result <- model$s3gen$inference(
                speech_tokens = speech_tokens,
                ref_dict = voice$ref_dict,
                finalize = TRUE,
                traced = traced
            )
            audio <- result[[1]]
        })
    }

    # Convert to numeric (moves to CPU)
    audio_samples <- as.numeric(audio$squeeze()$cpu())

    # Restore T3 and voice encoder to GPU for reuse
    if (grepl("^cuda", device)) {
        model$t3$to(device = device)
        model$voice_encoder$to(device = device)
    }

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
    cat("Chatterbox TTS Model\n")
    cat("  Device:", x$device, "\n")
    cat("  Loaded:", x$loaded, "\n")
    if (x$loaded) {
        cat("  Components:\n")
        cat("    - T3 (text-to-tokens)\n")
        cat("    - S3Gen (tokens-to-waveform)\n")
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
                       device = "cpu", autocast = NULL)
{
    # Create and load model (caches after first load)
    model <- chatterbox(device)
    model <- load_chatterbox(model)

    # Generate
    if (is.null(output_path)) {
        generate(model, text, reference_audio, autocast = autocast)
    } else {
        tts_to_file(model, text, reference_audio, output_path, autocast = autocast)
    }
}

