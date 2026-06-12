# chatterbox - High-level Text-to-Speech API
# Provides simple interface for TTS generation using the Chatterbox engine

# ============================================================================
# Text Normalization
# ============================================================================

#' Normalize text for TTS
#'
#' Lowercases words that contain internal capital letters (e.g.
#' "ALERT", "Rarely"). The Chatterbox model interprets internal capitals
#' as emphasis cues, which often causes it to produce only the first word
#' followed by silence. Sentence-initial capitals are left alone.
#'
#' @param text Character scalar.
#' @return Normalized text.
#' @export
normalize_tts_text <- function(text) {
    if (!is.character(text) || length(text) != 1L || is.na(text)) {
        return(text)
    }
    # Split into tokens preserving whitespace and punctuation
    parts <- strsplit(text, "(\\s+)", perl = TRUE)[[1]]
    if (length(parts) == 0L) {
        return(text)
    }

    # Track sentence boundary: first word, or word right after .!?
    prev_was_sentence_end <- TRUE
    out <- character(length(parts))
    for (i in seq_along(parts)) {
        word <- parts[i]
        letters_only <- gsub("[^A-Za-z]", "", word)
        is_capitalized <- nzchar(letters_only) &&
        grepl("^[A-Z]", letters_only)
        has_internal_caps <- nzchar(letters_only) &&
        grepl("[A-Z]", substring(letters_only, 2L))
        is_all_caps <- nzchar(letters_only) &&
        letters_only == toupper(letters_only) && nchar(letters_only) > 1L

        # Lowercase if:
        # - all caps (and longer than 1 letter, to skip "I"), OR
        # - internal caps (camelCase / weirdCase), OR
        # - capitalized mid-sentence (not first word and not after .!?),
        #   except for the standalone pronoun "I"
        is_pronoun_i <- letters_only == "I"
        should_lower <- is_all_caps || has_internal_caps ||
        (is_capitalized && !prev_was_sentence_end && !is_pronoun_i)
        if (should_lower) {
            out[i] <- tolower(word)
        } else {
            out[i] <- word
        }

        # Update sentence-end tracker for next word
        prev_was_sentence_end <- grepl("[.!?]\\s*$", word)
    }
    paste(out, collapse = " ")
}

# ============================================================================
# Chatterbox TTS Model
# ============================================================================

#' Create Chatterbox TTS model
#'
#' @param device Device to use ("cpu", "cuda", "mps", etc.)
#' @param turbo Use turbo model (GPT-2 backbone, MeanFlow decoder). Default FALSE.
#' @return Chatterbox TTS model object
#' @export
chatterbox <- function(device = "cpu", turbo = FALSE) {
    # Fall back to CPU when the requested accelerator is absent
    # (Python from_pretrained does the same for MPS)
    if (grepl("^cuda", device) && !torch::cuda_is_available()) {
        warning("CUDA requested but not available; using CPU", call. = FALSE)
        device <- "cpu"
    }
    if (device == "mps" && !torch::backends_mps_is_available()) {
        warning("MPS requested but not available; using CPU", call. = FALSE)
        device <- "cpu"
    }

    structure(
              list(device = device, turbo = turbo, t3 = NULL, s3gen = NULL,
                   voice_encoder = NULL, tokenizer = NULL, loaded = FALSE),
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
load_chatterbox <- function(model) {
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
    model$voice_encoder <- load_voice_encoder_weights(model$voice_encoder,
        ve_weights)
    rm(ve_weights) ; gc()
    model$voice_encoder$to(device = device)
    model$voice_encoder$eval()

    # Load T3 model
    message("Loading T3 text-to-speech model...")
    t3_weights <- read_safetensors(paths$t3_cfg, "cpu")
    model$t3 <- t3_model() # Creates T3Model instance
    model$t3 <- load_t3_weights(model$t3, t3_weights)
    rm(t3_weights) ; gc()
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
load_chatterbox_turbo <- function(model) {
    device <- model$device
    message("Loading Chatterbox Turbo model to ", device, "...")

    # Get turbo model file paths
    message("Loading turbo model files...")
    paths <- get_turbo_model_paths()

    # Load GPT-2 tokenizer
    message("Loading GPT-2 tokenizer...")
    model$tokenizer <- load_gpt2_tokenizer(paths$vocab, paths$merges,
        paths$added_tokens)

    # Load voice encoder (same as standard)
    message("Loading voice encoder...")
    ve_weights <- read_safetensors(paths$ve, "cpu")
    model$voice_encoder <- voice_encoder()
    model$voice_encoder <- load_voice_encoder_weights(model$voice_encoder, ve_weights)
    rm(ve_weights) ; gc()
    model$voice_encoder$to(device = device)
    model$voice_encoder$eval()

    # Load T3 turbo model (GPT-2 backbone)
    message("Loading T3 turbo model (GPT-2 backbone)...")
    t3_weights <- read_safetensors(paths$t3_turbo_v1, "cpu")
    model$t3 <- t3_model_turbo()
    model$t3 <- load_t3_turbo_weights(model$t3, t3_weights)
    rm(t3_weights) ; gc()
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
is_loaded <- function(model) {
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
#' @param norm_loudness Normalize the reference to -27 LUFS before
#'   conditioning (\code{\link{normalize_loudness}}). Default matches
#'   Python: \code{TRUE} for turbo models, \code{FALSE} for standard.
#' @return Voice embedding that can be used for synthesis
#' @export
create_voice_embedding <- function(model, audio, sample_rate = NULL,
                                   autocast = NULL, norm_loudness = NULL) {
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

    # Python parity (tts_turbo.py prepare_conditionals): turbo
    # normalizes the reference to -27 LUFS before any conditioning;
    # standard chatterbox does not.
    if (is.null(norm_loudness)) {
        norm_loudness <- isTRUE(model$turbo)
    }
    if (isTRUE(norm_loudness)) {
        samples <- normalize_loudness(samples, sample_rate)
    }

    # Resample to 16kHz for voice encoder and tokenizer
    if (sample_rate != 16000) {
        samples_16k <- resample_audio(samples, sample_rate, 16000)
    } else {
        samples_16k <- samples
    }

    # Python parity (tts.py prepare_conditionals): the S3Gen reference is
    # capped at 10 s (DEC_COND_LEN) and the tokenizer conditioning prompt
    # at 6 s (ENC_COND_LEN); the voice encoder sees the full reference.
    # Longer prompts than the model was trained on degrade quality and
    # blow up CFM attention cost.
    samples_dec <- samples[seq_len(min(length(samples),
                                       as.integer(10 * sample_rate)))]
    samples_16k_enc <- samples_16k[seq_len(min(length(samples_16k), 6L * 16000L))]

    # Convert to tensor
    audio_tensor <- torch::torch_tensor(samples_dec, dtype = torch::torch_float32())$unsqueeze(1)$to(device = device)
    audio_16k <- torch::torch_tensor(samples_16k, dtype = torch::torch_float32())$unsqueeze(1)$to(device = device)
    audio_16k_enc <- torch::torch_tensor(samples_16k_enc, dtype = torch::torch_float32())$unsqueeze(1)$to(device = device)

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
        tok_result <- model$s3gen$tokenizer$forward(audio_16k_enc, max_len = cond_prompt_len)
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
#' @param top_p Top-p (nucleus) sampling threshold. Default 1.0 (disabled),
#'   matching the Python reference.
#' @param min_p Minimum probability threshold relative to the most likely
#'   token (default 0.05, matching the Python reference). Standard model only.
#' @param autocast Use mixed precision (float16) on CUDA for faster
#'   inference. Default FALSE: the Python reference runs float32, and
#'   float16 output diverges slightly. Opt in for speed on tight VRAM.
#' @param traced Logical. Use JIT-traced inference. Default FALSE.
#' @param backend Character. Inference backend, either "r" or "jit".
#'   Default "r". The jit backend runs each token's full 30-layer
#'   forward as one TorchScript call (compiled once per session, in
#'   milliseconds, via \code{torch::jit_compile}): with tuned GC
#'   settings (see \code{\link{chatterbox_gc_options}}) it is the
#'   fastest native path (~11 ms/token long-form), auto-sizes its KV
#'   cache so generation always completes, and ships no compiled code.
#'   It replaced an equivalent C++ backend (within ~20\% of its speed)
#'   that required linking against torch's private libraries.
#' @param top_k Integer. Top-k sampling parameter (turbo model only).
#'   Default 1000.
#' @param repetition_penalty Numeric. Repetition penalty. Default 1.2.
#'   Applied sign-dependently like HF transformers: positive logits are
#'   divided, negative ones multiplied.
#' @param normalize_text Logical. If TRUE (default), pre-process text to
#'   reduce model failure modes: lowercase words with internal capitals
#'   (which the model interprets as emphasis cues and often produces
#'   silent audio for). Set to FALSE to skip case normalization.
#'   Punctuation normalization (whitespace collapse, first-letter
#'   capitalization, trailing period) always runs, matching the Python
#'   reference implementation.
#' @param max_new_tokens Maximum speech tokens to generate (default 1000,
#'   = 40 s of audio; the model's own ceiling is 4096).
#' @param max_cache_len KV cache positions for the jit and traced
#'   backends. Default NULL: jit auto-sizes so generation always fits
#'   (~1 MB VRAM per position); traced keeps its 350-position trace (a
#'   new size triggers a fresh ~50 s trace). Ignored by the pure-R
#'   backend, which has no pre-allocated cache.
#' @param skip_vocoder Logical. If TRUE, stop after flow matching and
#'   return the mel spectrogram instead of audio (Python 0.1.7's
#'   \code{skip_vocoder}). The result has a \code{mel} element (tensor,
#'   batch x 80 x frames; 50 frames/s) and no \code{audio}.
#' @return List with elements:
#'   \describe{
#'     \item{audio}{Numeric vector of audio samples (omitted when
#'       \code{skip_vocoder = TRUE}, which returns \code{mel} instead)}
#'     \item{sample_rate}{Sample rate in Hz}
#'     \item{eos_found}{Logical. Whether the model emitted an end-of-speech
#'       token (TRUE) or hit the token cap (FALSE). FALSE often indicates
#'       garbage output and a need to retry or split the input.}
#'     \item{n_tokens}{Number of speech tokens generated}
#'     \item{audio_sec}{Audio duration in seconds}
#'   }
#' @export
generate <- function(model, text, voice, exaggeration = 0.5,
                     cfg_weight = 0.5, temperature = 0.8, top_p = 1.0,
                     min_p = 0.05, autocast = NULL, traced = FALSE,
                     backend = c("r", "jit"), top_k = 1000L,
                     repetition_penalty = 1.2, normalize_text = TRUE,
                     max_new_tokens = 1000L, max_cache_len = NULL,
                     skip_vocoder = FALSE) {
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }

    if (isTRUE(normalize_text)) {
        text <- normalize_tts_text(text)
    }

    # Punctuation normalization, applied unconditionally to match the
    # Python reference (tts.py generate): collapses whitespace runs,
    # capitalizes the first letter, rewrites uncommon punctuation, and
    # appends a final period when the text ends without one. The missing
    # trailing period was a major cause of EOS-not-found failures.
    text <- punc_norm(text)

    device <- model$device
    is_turbo <- isTRUE(model$turbo)
    # Default OFF: the Python reference runs float32 everywhere (S3Gen's
    # fp16 flag is explicitly False upstream). Mixed precision is an
    # opt-in speed/VRAM trade.
    use_autocast <- isTRUE(autocast) && grepl("^cuda", device)

    # Handle voice input
    if (is.character(voice)) {
        voice <- create_voice_embedding(model, voice, autocast = use_autocast)
    } else if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object or path to reference audio")
    }

    speech_tokens <- .t3_text_to_tokens(model, text, voice,
                                        exaggeration = exaggeration, cfg_weight = cfg_weight,
                                        temperature = temperature, top_p = top_p, min_p = min_p,
                                        traced = traced, backend = backend, top_k = top_k,
                                        repetition_penalty = repetition_penalty,
                                        max_new_tokens = max_new_tokens, max_cache_len = max_cache_len,
                                        use_autocast = use_autocast)

    # Capture EOS status before drop_invalid_tokens strips the attribute
    eos_found <- isTRUE(attr(speech_tokens, "eos_found"))

    # Drop invalid tokens
    speech_tokens <- drop_invalid_tokens(speech_tokens)
    n_tokens <- as.integer(speech_tokens$size(1L))

    if (length(speech_tokens) == 0) {
        warning("No valid speech tokens generated")
        return(list(audio = numeric(0), sample_rate = S3GEN_SR,
                    eos_found = eos_found, n_tokens = 0L, audio_sec = 0))
    }

    if (!eos_found) {
        warning("Generation hit token cap without emitting end-of-speech ",
                "(", n_tokens, " tokens). Output may be garbage; ",
                "consider splitting the input or retrying.")
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

    # Generate waveform (or just the mel) with S3Gen
    message(if (skip_vocoder) "Synthesizing mel spectrogram..."
        else "Synthesizing waveform...")
    if (is_turbo) {
        n_cfm_steps <- 2L
    } else {
        n_cfm_steps <- NULL
    }

    if (use_autocast) {
        torch::with_autocast(device_type = "cuda", {
            torch::with_no_grad({
                result <- model$s3gen$inference(
                    speech_tokens = speech_tokens,
                    ref_dict = voice$ref_dict,
                    finalize = TRUE,
                    traced = traced,
                    n_cfm_timesteps = n_cfm_steps,
                    skip_vocoder = skip_vocoder
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
                n_cfm_timesteps = n_cfm_steps,
                skip_vocoder = skip_vocoder
            )
            audio <- result[[1]]
        })
    }

    if (skip_vocoder) {
        # audio here is the mel (batch, 80, frames); 50 frames/s
        audio_sec <- audio$size(3) / 50
        message("Done! Generated mel for ", round(audio_sec, 2),
                " seconds of audio (vocoder skipped).")
        return(list(
                    mel = audio,
                    sample_rate = S3GEN_SR,
                    eos_found = eos_found,
                    n_tokens = n_tokens,
                    audio_sec = audio_sec
            ))
    }

    # Convert to numeric
    audio_samples <- as.numeric(audio$squeeze()$cpu())
    audio_sec <- length(audio_samples) / S3GEN_SR

    message("Done! Generated ", round(audio_sec, 2), " seconds of audio.")

    list(
         audio = audio_samples,
         sample_rate = S3GEN_SR,
         eos_found = eos_found,
         n_tokens = n_tokens,
         audio_sec = audio_sec
    )
}

#' T3 stage shared by generate() and generate_batch(): tokenize one
#' text and run the configured inference backend. \code{voice} must
#' already be a voice_embedding. Returns the speech tokens with the
#' eos_found attribute intact.
#'
#' @noRd
.t3_text_to_tokens <- function(model, text, voice, exaggeration, cfg_weight,
                               temperature, top_p, min_p, traced, backend,
                               top_k, repetition_penalty, max_new_tokens,
                               max_cache_len, use_autocast) {
    device <- model$device
    is_turbo <- isTRUE(model$turbo)

    # Tokenize text
    if (is_turbo) {
        text_ids <- tokenize_text_gpt2(model$tokenizer, text)
        text_tokens <- torch::torch_tensor(text_ids,
            dtype = torch::torch_long())$unsqueeze(1L)$to(device = device)
    } else {
        text_tokens <- tokenize_text(model$tokenizer, text)
        text_tokens <- torch::torch_tensor(text_tokens,
            dtype = torch::torch_long())$unsqueeze(1L)$to(device = device)
    }

    # Create T3 conditioning
    cond <- t3_cond(
                    speaker_emb = voice$ve_embedding,
                    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
                    emotion_adv = if (is_turbo) NULL else exaggeration
    )

    message("Generating speech tokens...")

    if (is_turbo) {
        # Turbo inference: no CFG, no min_p, uses top_k
        inf_args <- list(
                         model = model$t3,
                         cond = cond,
                         text_tokens = text_tokens,
                         temperature = temperature,
                         top_k = top_k,
                         top_p = top_p,
                         repetition_penalty = repetition_penalty
        )
        inference_fn <- t3_inference_turbo
    } else {
        # Standard inference with CFG
        backend <- match.arg(backend, c("r", "jit"))
        if (backend == "jit") {
            inference_fn <- t3_inference_jit
        } else if (traced) {
            inference_fn <- t3_inference_traced
        } else {
            inference_fn <- t3_inference
        }

        inf_args <- list(
                         model = model$t3,
                         cond = cond,
                         text_tokens = text_tokens,
                         cfg_weight = cfg_weight,
                         temperature = temperature,
                         top_p = top_p,
                         min_p = min_p,
                         repetition_penalty = repetition_penalty,
                         max_new_tokens = max_new_tokens
        )
        # Cache sizing only applies to the pre-allocated-cache backends;
        # jit auto-sizes when NULL, traced keeps its 350 default (a new
        # size means a fresh ~50s JIT trace)
        if (!is.null(max_cache_len) && (backend == "jit" || traced)) {
            inf_args$max_cache_len <- max_cache_len
        }
    }

    if (use_autocast) {
        torch::with_autocast(device_type = "cuda", {
            torch::with_no_grad({
                speech_tokens <- do.call(inference_fn, inf_args)
            })
        })
    } else {
        torch::with_no_grad({
            speech_tokens <- do.call(inference_fn, inf_args)
        })
    }
    speech_tokens
}

#' Generate speech for several texts with one batched synthesis pass
#'
#' Runs T3 token generation per text (autoregressive, sequential), then
#' synthesizes ALL utterances in a single batched S3Gen pass (one CFM
#' solve and one vocoder call over the padded batch). Per-utterance
#' results match single \code{\link{generate}} calls up to CFM noise
#' handling - the fixed noise buffer means row i sees the same initial
#' noise it would alone. Standard model only.
#'
#' @param model Loaded chatterbox model (standard, not turbo)
#' @param texts Character vector of texts to synthesize
#' @param voice Shared voice: voice_embedding or reference audio path
#' @param ... Arguments passed through to the T3 stage, as in
#'   \code{\link{generate}} (exaggeration, cfg_weight, temperature,
#'   top_p, min_p, backend, repetition_penalty, normalize_text,
#'   max_new_tokens, max_cache_len)
#' @return List with one \code{\link{generate}}-style result per text
#'   (audio, sample_rate, eos_found, n_tokens, audio_sec)
#' @export
generate_batch <- function(model, texts, voice, ...) {
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }
    if (isTRUE(model$turbo)) {
        stop("generate_batch supports the standard model only")
    }
    if (!is.character(texts) || length(texts) == 0) {
        stop("texts must be a non-empty character vector")
    }

    args <- list(...)
    arg_or <- function(name, default) args[[name]] %||% default
    normalize_text <- isTRUE(arg_or("normalize_text", TRUE))
    use_autocast <- isTRUE(arg_or("autocast", FALSE)) &&
    grepl("^cuda", model$device)

    if (is.character(voice)) {
        voice <- create_voice_embedding(model, voice)
    } else if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object or path to ",
             "reference audio")
    }

    # T3 per text (autoregressive generation does not batch; lengths
    # and EOS differ per utterance)
    token_vecs <- vector("list", length(texts))
    eos <- logical(length(texts))
    for (i in seq_along(texts)) {
        txt <- texts[i]
        if (normalize_text) {
            txt <- normalize_tts_text(txt)
        }
        txt <- punc_norm(txt)
        tokens <- .t3_text_to_tokens(model, txt, voice,
                                     exaggeration = arg_or("exaggeration", 0.5),
                                     cfg_weight = arg_or("cfg_weight", 0.5),
                                     temperature = arg_or("temperature", 0.8),
                                     top_p = arg_or("top_p", 1.0),
                                     min_p = arg_or("min_p", 0.05),
                                     traced = isTRUE(arg_or("traced", FALSE)),
                                     backend = arg_or("backend", "r"),
                                     top_k = arg_or("top_k", 1000L),
                                     repetition_penalty = arg_or("repetition_penalty", 1.2),
                                     max_new_tokens = arg_or("max_new_tokens", 1000L),
                                     max_cache_len = arg_or("max_cache_len", NULL),
                                     use_autocast = use_autocast)
        eos[i] <- isTRUE(attr(tokens, "eos_found"))
        token_vecs[[i]] <- as.integer(drop_invalid_tokens(tokens))
        if (!eos[i]) {
            warning("Text ", i, " hit the token cap without end-of-speech (",
                    length(token_vecs[[i]]), " tokens). Output may be garbage.",
                    call. = FALSE)
        }
    }

    lens <- vapply(token_vecs, length, integer(1))
    results <- vector("list", length(texts))
    empty <- lens == 0L
    for (i in which(empty)) {
        warning("No valid speech tokens for text ", i, call. = FALSE)
        results[[i]] <- list(audio = numeric(0), sample_rate = S3GEN_SR,
                             eos_found = eos[i], n_tokens = 0L, audio_sec = 0)
    }
    live <- which(!empty)
    if (length(live) == 0) {
        return(results)
    }

    # Pad to a (B, Tmax) batch; padded tail is masked by
    # speech_token_lens all the way through CFM and trimmed after the
    # vocoder
    t_max <- max(lens[live])
    mat <- t(vapply(token_vecs[live],
                    function(v) c(v, rep(0L, t_max - length(v))), integer(t_max)))
    speech_tokens <- torch::torch_tensor(mat,
        dtype = torch::torch_long())$to(device = model$device)

    message("Synthesizing ", length(live), " waveforms in one batch...")
    torch::with_no_grad({
        out <- model$s3gen$inference(
                                     speech_tokens = speech_tokens,
                                     ref_dict = voice$ref_dict,
                                     finalize = TRUE,
                                     speech_token_lens = lens[live]
        )
    })
    wavs <- out[[1]]$cpu()

    # 2 mel frames per token, 480 samples per mel frame
    for (k in seq_along(live)) {
        i <- live[k]
        n_samples <- lens[i] * 2L * 480L
        audio <- as.numeric(wavs[k, 1:min(n_samples, wavs$size(2))])
        results[[i]] <- list(
                             audio = audio,
                             sample_rate = S3GEN_SR,
                             eos_found = eos[i],
                             n_tokens = lens[i],
                             audio_sec = length(audio) / S3GEN_SR
        )
    }
    results
}

#' Generate speech and save to file
#'
#' @param model Chatterbox model
#' @param text Text to synthesize
#' @param voice Voice embedding or path to reference audio
#' @param output_path Output file path (WAV format)
#' @param ... Additional arguments passed to generate()
#' @return Invisibly returns a list with elements: \code{path},
#'   \code{eos_found}, \code{n_tokens}, \code{audio_sec}. When iterating
#'   over many texts, collect these into a data.frame to identify which
#'   inputs failed (\code{eos_found = FALSE}) and need reprocessing.
#' @export
tts_to_file <- function(model, text, voice, output_path, ...) {
    if (isTRUE(list(...)$skip_vocoder)) {
        stop("skip_vocoder makes no sense here: there is no audio to ",
             "write. Use generate() to get the mel.")
    }
    result <- generate(model, text, voice, ...)
    write_audio(result$audio, result$sample_rate, output_path)
    invisible(list(path = output_path, eos_found = isTRUE(result$eos_found),
                   n_tokens = result$n_tokens %||% NA_integer_,
                   audio_sec = result$audio_sec %||% NA_real_))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ============================================================================
# Streaming TTS (for longer texts)
# ============================================================================

#' Split text into TTS-sized chunks
#'
#' Sentences first; sentences longer than chunk_size chars are further
#' split at comma boundaries, packed greedily. A clause with no commas
#' stays whole (splitting mid-clause hurts prosody more than a long
#' generation does).
#'
#' @param text Input text
#' @param chunk_size Target maximum chunk length in characters
#' @return Character vector of chunks
#' @noRd
.split_text_chunks <- function(text, chunk_size = 200L) {
    sentences <- strsplit(text, "(?<=[.!?])\\s+", perl = TRUE)[[1]]
    chunks <- character(0)
    for (s in sentences) {
        if (nchar(s) <= chunk_size) {
            chunks <- c(chunks, s)
            next
        }
        parts <- strsplit(s, "(?<=,)\\s+", perl = TRUE)[[1]]
        cur <- ""
        for (p in parts) {
            if (nzchar(cur) && nchar(cur) + 1L + nchar(p) > chunk_size) {
                chunks <- c(chunks, cur)
                cur <- p
            } else {
                if (nzchar(cur)) {
                    cur <- paste(cur, p)
                } else {
                    cur <- p
                }
            }
        }
        if (nzchar(cur)) {
            chunks <- c(chunks, cur)
        }
    }
    chunks[nzchar(trimws(chunks))]
}

#' Generate speech in chunks (for long texts)
#'
#' Splits at sentence boundaries; sentences longer than chunk_size
#' characters are further split at commas. Collects garbage once per
#' chunk (see \code{\link{chatterbox_gc_options}}).
#'
#' @param model Chatterbox model
#' @param text Text to synthesize
#' @param voice Voice embedding
#' @param chunk_size Maximum characters per chunk (default 200)
#' @param ... Additional arguments passed to generate()
#' @return List with audio and sample_rate
#' @export
tts_chunked <- function(model, text, voice, chunk_size = 200, ...) {
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }

    # Sentences, with oversized sentences subdivided at commas
    # (chunk_size was previously accepted but never used: run-on
    # sentences passed through whole and hit backend token caps)
    sentences <- .split_text_chunks(text, chunk_size)

    all_audio <- numeric(0)

    for (i in seq_along(sentences)) {
        sentence <- sentences[i]
        message(sprintf("Processing chunk %d/%d: %s...", i,
                        length(sentences), substr(sentence, 1, 50)))

        result <- generate(model, sentence, voice, ...)
        all_audio <- c(all_audio, result$audio)

        # Collect once per utterance: frees the chunk's dead tensor
        # handles (and their GPU memory) in one pass instead of letting
        # torch's allocator conscript thousands of mid-loop collections.
        # See ?chatterbox_gc_options.
        gc(verbose = FALSE)
    }

    list(audio = all_audio, sample_rate = S3GEN_SR)
}

# ============================================================================
# Print Methods
# ============================================================================

#' Print method for chatterbox
#'
#' @param x Chatterbox model
#' @param ... Ignored
#' @export
print.chatterbox <- function(x, ...) {
    if (isTRUE(x$turbo)) {
        variant <- "Turbo"
    } else {
        variant <- "Standard"
    }
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
print.voice_embedding <- function(x, ...) {
    cat("Voice Embedding\n")
    cat("  Speaker embedding shape:",
        paste(dim(x$ve_embedding), collapse = " x "), "\n")
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
#' @param turbo Logical. Use turbo architecture. Default FALSE.
#' @return If output_path is NULL, returns list with audio and sample_rate.
#'         Otherwise writes to file and returns path invisibly.
#' @export
quick_tts <- function(text, reference_audio, output_path = NULL,
                      device = "cpu", autocast = NULL, turbo = FALSE) {
    # Create and load model (caches after first load)
    model <- chatterbox(device, turbo = turbo)
    model <- load_chatterbox(model)

    # Generate
    if (is.null(output_path)) {
        generate(model, text, reference_audio, autocast = autocast)
    } else {
        tts_to_file(model, text, reference_audio, output_path,
                    autocast = autocast)
    }
}
