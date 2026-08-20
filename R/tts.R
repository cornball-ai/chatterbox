# chatterbox - High-level Text-to-Speech API
# Provides simple interface for TTS generation using the Chatterbox engine

# ============================================================================
# Text Normalization
# ============================================================================

#' Lowercase internal-capital words (R-specific mitigation)
#'
#' Lowercases words that contain internal capital letters (e.g.
#' "ALERT", "Rarely"). The Chatterbox model interprets internal capitals
#' as emphasis cues, which often causes it to produce only the first word
#' followed by silence. Sentence-initial capitals are left alone. Not part
#' of the Python reference (which has only punc_norm); whether it is still
#' needed after the column-major/STFT fixes is an empirical question.
#'
#' @param text Character scalar.
#' @return Text with internal-capital words lowercased.
#' @noRd
normalize_internal_caps <- function(text) {
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

#' Normalize text for TTS
#'
#' The single normalization entry point. Applies, in order: the
#' R-specific internal-caps mitigation (\code{normalize_internal_caps}),
#' then punctuation normalization (\code{punc_norm}: whitespace collapse,
#' first-letter capitalization, uncommon-punctuation rewrite, trailing
#' period). \code{punc_norm} is the Python-parity piece; the caps step is
#' R-only and can be turned off.
#'
#' @param text Character scalar.
#' @param caps Apply the internal-caps mitigation. Default TRUE.
#' @param punctuation Apply punctuation normalization. Default TRUE.
#' @return Normalized text.
#' @examples
#' normalize_tts_text("hello   world")
#' @export
normalize_tts_text <- function(text, caps = TRUE, punctuation = TRUE) {
    if (isTRUE(caps)) {
        text <- normalize_internal_caps(text)
    }
    if (isTRUE(punctuation)) {
        text <- punc_norm(text)
    }
    text
}

# ============================================================================
# Chatterbox TTS Model
# ============================================================================

#' Create (and load) a Chatterbox TTS model
#'
#' Constructs the model object and, by default, loads the pretrained
#' weights in the same call - the Python reference's
#' \code{from_pretrained}/\code{from_local} do both at once. Pass
#' \code{load = FALSE} for the bare object (e.g. to inspect it or test
#' the not-loaded error paths), then load later with
#' \code{\link{load_chatterbox}}.
#'
#' @details
#' When \code{tune_gc = TRUE} (the default) and \code{device} is CUDA, this
#' raises torch's allocator GC floors before the first CUDA op. torch otherwise
#' runs \code{gc()} on nearly every allocation once a model occupies more than
#' 20\% of VRAM, which dominates inference. It sets session-global
#' \code{torch.cuda_allocator_reserved_rate} (the model footprint over VRAM) and
#' \code{torch.threshold_call_gc}, only when they are unset, so an explicit
#' setting always wins. This is a deliberate, persistent side effect (torch
#' reads the rates later, at CUDA init); pass \code{tune_gc = FALSE} to skip it.
#'
#' @param device Device to use ("cpu", "cuda", "mps", etc.)
#' @param turbo Use turbo model (GPT-2 backbone, MeanFlow decoder). Default FALSE.
#' @param load Load pretrained weights before returning. Default TRUE.
#'   Requires a prior download (\code{\link{download_chatterbox_models}}).
#' @param tune_gc Tune torch's CUDA GC rates for faster inference (CUDA only,
#'   and only when unset). Persistent session side effect; default TRUE. See
#'   Details.
#' @return Chatterbox TTS model object, loaded unless \code{load = FALSE}
#' @examples
#' \dontrun{
#' # Construct and load the standard model on GPU
#' model <- chatterbox("cuda")
#'
#' # Bare object without weights (load later with load_chatterbox())
#' model <- chatterbox("cuda", load = FALSE)
#' }
#' @export
chatterbox <- function(device = "cpu", turbo = FALSE, load = TRUE,
                       tune_gc = TRUE) {
    # GC tuning must run before the first CUDA op (cuda_is_available below):
    # torch reads its allocator GC rates once, at lazy CUDA init.
    if (isTRUE(tune_gc)) {
        .set_cuda_gc_options(device, turbo)
    }
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

    model <- structure(
                       list(device = device, turbo = turbo, t3 = NULL, s3gen = NULL,
                            voice_encoder = NULL, tokenizer = NULL, loaded = FALSE),
                       class = "chatterbox"
    )
    if (isTRUE(load)) {
        model <- load_chatterbox(model)
    }
    model
}

# Tune torch's CUDA garbage collection so it doesn't collect on nearly every
# allocation. torch reads these rates ONCE, at lazy CUDA init (the first CUDA
# op), so they must be set before then - this runs at the top of chatterbox(),
# ahead of cuda_is_available(). The default reserved-rate floor (0.2) makes
# torch gc on each allocation once a model occupies more than 20% of VRAM,
# which dominated inference (53% of wall time was gc, the GPU work ~13%). The
# floor is the model's footprint as a fraction of VRAM: gc stays off until
# reserved memory exceeds the model itself. Footprints (fp32) are 4.1GB
# regular and 3.6GB turbo, so e.g. a 16GB card gives 0.26 / 0.23 and a 6GB
# card gives 0.68 / 0.60. threshold_call_gc (host MB per forced gc) is raised
# off its 4GB default too. Both respect an explicit user override. See torch's
# memory-management vignette (torch.cuda_allocator_reserved_rate).
.set_cuda_gc_options <- function(device, turbo = FALSE) {
    if (is.null(device) || !grepl("^cuda", device)) {
        return(invisible(NULL))
    }
    if (is.null(getOption("torch.threshold_call_gc"))) {
        options(torch.threshold_call_gc = 16000)
    }
    if (!is.null(getOption("torch.cuda_allocator_reserved_rate"))) {
        return(invisible(NULL))
    }
    idx <- suppressWarnings(as.integer(sub("^cuda:?", "", device)))
    if (length(idx) != 1L || is.na(idx)) {
        idx <- 0L
    }
    total_gb <- tryCatch(
                         as.numeric(system2("nvidia-smi",
                c("--query-gpu=memory.total", "--format=csv,noheader,nounits",
                    paste0("--id=", idx)), stdout = TRUE)[1]) / 1024,
                         error = function(e) NA_real_)
    if (is.na(total_gb) || total_gb <= 0) {
        return(invisible(NULL))
    }
    if (isTRUE(turbo)) {
        footprint_gb <- 3.6
    } else {
        footprint_gb <- 4.1
    }
    rate <- min(0.92, max(0.20, footprint_gb / total_gb))
    options(torch.cuda_allocator_reserved_rate = rate)
    message(sprintf(
                    "chatterbox: torch.cuda_allocator_reserved_rate = %.2f, threshold_call_gc = %d MB (%s model, %.0f GB VRAM)",
                    rate, getOption("torch.threshold_call_gc"),
            if (isTRUE(turbo)) "turbo 3.6GB" else "regular 4.1GB", total_gb))
    invisible(rate)
}

#' Load Chatterbox model weights
#'
#' Load pretrained weights for all model components.
#' Requires prior download via \code{\link{download_chatterbox_models}}.
#' Idempotent: an already-loaded model is returned unchanged, so
#' \code{chatterbox(load = TRUE)} followed by a stray
#' \code{load_chatterbox()} does not reload.
#'
#' @param model Chatterbox model object
#' @return Chatterbox model with loaded weights
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda", load = FALSE)
#' model <- load_chatterbox(model)
#' }
#' @export
load_chatterbox <- function(model) {
    if (!inherits(model, "chatterbox")) {
        stop("model must be a chatterbox object")
    }
    if (isTRUE(model$loaded)) {
        return(model)
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
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda", turbo = TRUE, load = FALSE)
#' model <- load_chatterbox_turbo(model)
#' }
#' @export
load_chatterbox_turbo <- function(model) {
    if (isTRUE(model$loaded)) {
        return(model)
    }
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
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' voice <- create_voice_embedding(model, "reference_voice.wav")
#' res <- generate(model, "Reusing a cached voice.", voice)
#' }
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
#' @param normalize_text Logical. Apply the R-specific internal-caps
#'   mitigation (lowercase words with internal capitals). Default FALSE:
#'   it addressed a "first word then silence" failure that was actually
#'   the column-major/STFT bug, now fixed - with the model corrected,
#'   leaving caps intact also preserves intended emphasis (ALL CAPS reads
#'   as emphasis) and acronyms. Set TRUE only if specific text misbehaves.
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
#' @param cfm_len Optional explicit traced-CFM length (the padded mel
#'   sequence, = 640 + 2 * tokens). Default NULL: the standard traced path
#'   sizes it from the tokens actually generated, rounded up to the
#'   250/500/1000 bucket ladder, so a slow speaker is covered without
#'   guessing from text length. Pass a value to pin it (e.g. to pre-trace
#'   a bucket). Ignored when not traced or for turbo.
#' @param skip_vocoder Logical. If TRUE, stop after flow matching and
#'   return the mel spectrogram instead of audio (Python 0.1.7's
#'   \code{skip_vocoder}). The result has a \code{mel} element (tensor,
#'   batch x 80 x frames; 50 frames/s) and no \code{audio}.
#' @param output_path Optional WAV path. When set, the audio is also
#'   written there (as a side effect) and the returned list gains a
#'   \code{path} element; the audio is still returned in full. Incompatible
#'   with \code{skip_vocoder} (no audio to write). Default NULL.
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
#'     \item{path}{Output file path (only when \code{output_path} is set)}
#'   }
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' res <- generate(model, "Hello world!", "reference_voice.wav")
#' write_audio(res$audio, res$sample_rate, "hello.wav")
#'
#' # Fastest native path: TorchScript decode loop
#' res <- generate(model, "Hello world!", "reference_voice.wav",
#'                 backend = "jit")
#' }
#' @export
generate <- function(model, text, voice, exaggeration = 0.5,
                     cfg_weight = 0.5, temperature = 0.8, top_p = 1.0,
                     min_p = 0.05, autocast = NULL, traced = FALSE,
                     backend = c("r", "jit"), top_k = 1000L,
                     repetition_penalty = 1.2, normalize_text = FALSE,
                     max_new_tokens = 1000L, max_cache_len = NULL,
                     cfm_len = NULL, skip_vocoder = FALSE, output_path = NULL) {
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }
    if (!is.null(output_path) && isTRUE(skip_vocoder)) {
        stop("output_path and skip_vocoder are incompatible: there is no ",
             "audio to write. Drop output_path to get the mel.", call. = FALSE)
    }

    # Single normalization pass. punctuation (punc_norm) always runs to
    # match the Python reference - whitespace collapse, first-letter
    # capitalization, uncommon-punctuation rewrite, trailing period (a
    # missing one was a major EOS-not-found cause). caps is the R-only
    # internal-caps mitigation, gated by normalize_text.
    text <- normalize_tts_text(text, caps = normalize_text, punctuation = TRUE)

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

    # Size the traced CFM to the tokens we actually produced, rounded up
    # to the bucket ladder (250/500/1000) so only a few CFM sizes are ever
    # traced. Exact post-T3 sizing replaces guessing from text length: a
    # slow speaker emits more tokens and is covered automatically. An
    # explicit cfm_len pins it instead (the serve warmup pre-traces each
    # bucket that way). Standard traced path only; turbo uses MeanFlow.
    if (!is_turbo && isTRUE(traced)) {
        options(chatterbox.cfm_len = if (!is.null(cfm_len)) {
                as.integer(cfm_len)
            } else {
                640L + 2L * .token_bucket(n_tokens)
            })
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

    out <- list(
                audio = audio_samples,
                sample_rate = S3GEN_SR,
                eos_found = eos_found,
                n_tokens = n_tokens,
                audio_sec = audio_sec
    )
    if (!is.null(output_path)) {
        write_audio(audio_samples, S3GEN_SR, output_path)
        message("Wrote ", output_path)
        out$path <- output_path
    }
    out
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
    } else {
        text_ids <- tokenize_text(model$tokenizer, text)
    }

    # Guard the input text-token limit. T3 uses a learned positional
    # embedding table sized max_text_tokens + 2; longer input indexes
    # past it (a cryptic crash) rather than truncating, and neither R nor
    # the Python reference warns. Fail loudly so the caller splits the
    # text (see tts_chunked) instead of getting a stack trace.
    max_text <- model$t3$config$max_text_tokens
    if (!is.null(max_text) && length(text_ids) > max_text) {
        stop("Input text is too long: ", length(text_ids), " text tokens ",
             "exceed the T3 limit of ", max_text, ". Split the text before ",
             "generate() (e.g. tts_chunked).", call. = FALSE)
    }

    text_tokens <- torch::torch_tensor(text_ids,
                                       dtype = torch::torch_long())$unsqueeze(1L)$to(device = device)

    # Create T3 conditioning
    cond <- t3_cond(
                    speaker_emb = voice$ve_embedding,
                    cond_prompt_speech_tokens = voice$cond_prompt_speech_tokens,
                    emotion_adv = if (is_turbo) NULL else exaggeration
    )

    message("Generating speech tokens...")

    if (is_turbo) {
        # Turbo inference: no CFG, no min_p, uses top_k. backend = "jit"
        # runs the GPT-2 TorchScript decode step (faster); "r" the eager
        # path.
        backend <- match.arg(backend, c("r", "jit"))
        inf_args <- list(
                         model = model$t3,
                         cond = cond,
                         text_tokens = text_tokens,
                         temperature = temperature,
                         top_k = top_k,
                         top_p = top_p,
                         repetition_penalty = repetition_penalty,
                         max_new_tokens = max_new_tokens
        )
        if (backend == "jit") {
            inference_fn <- t3_inference_turbo_jit
            if (!is.null(max_cache_len)) {
                inf_args$max_cache_len <- max_cache_len
            }
        } else {
            inference_fn <- t3_inference_turbo
        }
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

#' T3 stage for several texts: normalize, tokenize and run T3 per text
#'
#' Autoregressive generation does not batch (lengths and EOS differ per
#' utterance), so this loops. \code{voice} must already be a
#' voice_embedding. Returns 0-indexed integer token vectors (invalid
#' tokens dropped) and per-text eos flags.
#' @noRd
.texts_to_speech_tokens <- function(model, texts, voice, normalize_text,
                                    use_autocast, exaggeration, cfg_weight,
                                    temperature, top_p, min_p, traced,
                                    backend, top_k, repetition_penalty,
                                    max_new_tokens, max_cache_len) {
    token_vecs <- vector("list", length(texts))
    eos <- logical(length(texts))
    for (i in seq_along(texts)) {
        txt <- normalize_tts_text(texts[i], caps = normalize_text,
                                  punctuation = TRUE)
        tokens <- .t3_text_to_tokens(model, txt, voice,
                                     exaggeration = exaggeration,
                                     cfg_weight = cfg_weight,
                                     temperature = temperature, top_p = top_p,
                                     min_p = min_p, traced = traced,
                                     backend = backend, top_k = top_k,
                                     repetition_penalty = repetition_penalty,
                                     max_new_tokens = max_new_tokens,
                                     max_cache_len = max_cache_len,
                                     use_autocast = use_autocast)
        eos[i] <- isTRUE(attr(tokens, "eos_found"))
        token_vecs[[i]] <- as.integer(drop_invalid_tokens(tokens))
        if (!eos[i]) {
            warning("Text ", i, " hit the token cap without end-of-speech (",
                    length(token_vecs[[i]]), " tokens). Output may be garbage.",
                    call. = FALSE)
        }
    }
    list(token_vecs = token_vecs, eos = eos)
}

#' S3Gen stage: synthesize already-generated speech tokens
#'
#' One utterance with \code{traced = TRUE} takes the fast traced-CFM path
#' sized from its actual token count; two or more run as a single eager
#' batched solve (traced CFM is fixed at batch 1). \code{token_vecs} are
#' 0-indexed integer vectors; empty ones yield empty results. Returns one
#' \code{\link{generate}}-style result list per input.
#' @noRd
.s3gen_batch_from_tokens <- function(model, token_vecs, eos, voice,
                                     traced = FALSE) {
    lens <- vapply(token_vecs, length, integer(1))
    results <- vector("list", length(token_vecs))
    for (i in which(lens == 0L)) {
        warning("No valid speech tokens for text ", i, call. = FALSE)
        results[[i]] <- list(audio = numeric(0), sample_rate = S3GEN_SR,
                             eos_found = eos[i], n_tokens = 0L, audio_sec = 0)
    }
    live <- which(lens > 0L)
    if (length(live) == 0L) {
        return(results)
    }
    mk_result <- function(i, audio) {
        list(audio = audio, sample_rate = S3GEN_SR, eos_found = eos[i],
             n_tokens = lens[i], audio_sec = length(audio) / S3GEN_SR)
    }

    if (length(live) == 1L && isTRUE(traced)) {
        # Single utterance: traced CFM, sized from the actual token count.
        i <- live
        options(chatterbox.cfm_len = 640L + 2L * .token_bucket(lens[i]))
        st <- torch::torch_tensor(matrix(token_vecs[[i]], nrow = 1L),
                                  dtype = torch::torch_long())$to(device = model$device)
        torch::with_no_grad({
            out <- model$s3gen$inference(speech_tokens = st,
                ref_dict = voice$ref_dict,
                finalize = TRUE, traced = TRUE)
        })
        results[[i]] <- mk_result(i, as.numeric(out[[1]]$squeeze()$cpu()))
        return(results)
    }

    # Two or more: one eager batched solve. Pad to (B, Tmax); the padded
    # tail is masked by speech_token_lens through CFM and trimmed after.
    t_max <- max(lens[live])
    mat <- t(vapply(token_vecs[live],
                    function(v) c(v, rep(0L, t_max - length(v))),
                    integer(t_max)))
    speech_tokens <- torch::torch_tensor(mat,
        dtype = torch::torch_long())$to(device = model$device)
    message("Synthesizing ", length(live), " waveforms in one batch...")
    torch::with_no_grad({
        out <- model$s3gen$inference(speech_tokens = speech_tokens,
                                     ref_dict = voice$ref_dict,
                                     finalize = TRUE,
                                     speech_token_lens = lens[live])
    })
    wavs <- out[[1]]$cpu()
    for (k in seq_along(live)) {
        i <- live[k]
        n_samples <- lens[i] * 2L * 480L
        results[[i]] <- mk_result(i,
                                  as.numeric(wavs[k, 1:min(n_samples, wavs$size(2))]))
    }
    results
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
#'   max_new_tokens, max_cache_len). \code{traced} and \code{autocast}
#'   affect the T3 stage only: the batched S3Gen synthesis always runs
#'   eager float32 (traced CFM is fixed at batch 1). The CFM trace-bucket
#'   sizing used by \code{\link{generate}} therefore does not apply here -
#'   batched S3Gen pads dynamically to the batch's longest utterance.
#' @return List with one \code{\link{generate}}-style result per text
#'   (audio, sample_rate, eos_found, n_tokens, audio_sec)
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' res <- generate_batch(model,
#'                       c("First sentence.", "Second sentence."),
#'                       "reference_voice.wav")
#' write_audio(res[[1]]$audio, res[[1]]$sample_rate, "first.wav")
#' }
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
    known <- c("exaggeration", "cfg_weight", "temperature", "top_p",
               "min_p", "traced", "backend", "top_k", "repetition_penalty",
               "normalize_text", "max_new_tokens", "max_cache_len",
               "autocast")
    unknown <- setdiff(names(args), known)
    if (length(unknown) > 0) {
        stop("Unsupported arguments: ", paste(unknown, collapse = ", "),
             ". generate_batch() accepts: ", paste(known, collapse = ", "))
    }
    arg_or <- function(name, default) args[[name]] %||% default
    use_autocast <- isTRUE(arg_or("autocast", FALSE)) &&
    grepl("^cuda", model$device)

    if (is.character(voice)) {
        voice <- create_voice_embedding(model, voice)
    } else if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object or path to ",
             "reference audio")
    }

    tk <- .texts_to_speech_tokens(model, texts, voice,
                                  normalize_text = isTRUE(arg_or("normalize_text", FALSE)),
                                  use_autocast = use_autocast,
                                  exaggeration = arg_or("exaggeration", 0.5),
                                  cfg_weight = arg_or("cfg_weight", 0.5),
                                  temperature = arg_or("temperature", 0.8),
                                  top_p = arg_or("top_p", 1.0), min_p = arg_or("min_p", 0.05),
                                  traced = isTRUE(arg_or("traced", FALSE)),
                                  backend = arg_or("backend", "r"), top_k = arg_or("top_k", 1000L),
                                  repetition_penalty = arg_or("repetition_penalty", 1.2),
                                  max_new_tokens = arg_or("max_new_tokens", 1000L),
                                  max_cache_len = arg_or("max_cache_len", NULL))

    # generate_batch is the throughput path: one eager batched S3Gen solve
    # (no traced single-utterance special case).
    .s3gen_batch_from_tokens(model, tk$token_vecs, tk$eos, voice,
                             traced = FALSE)
}

#' Generate speech and save to file
#'
#' Thin convenience wrapper over \code{\link{generate}} with
#' \code{output_path} set, kept for the file-summary return shape. New
#' code can call \code{generate(..., output_path = path)} directly.
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
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' tts_to_file(model, "Hello world!", "reference_voice.wav", "out.wav")
#' }
#' @export
tts_to_file <- function(model, text, voice, output_path, ...) {
    result <- generate(model, text, voice, ..., output_path = output_path)
    invisible(list(path = output_path, eos_found = isTRUE(result$eos_found),
                   n_tokens = result$n_tokens %||% NA_integer_,
                   audio_sec = result$audio_sec %||% NA_real_))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ============================================================================
# Streaming TTS (for longer texts)
# ============================================================================

#' Round a speech-token count up to the CFM trace ladder
#'
#' The traced CFM compiles (and pads to) a fixed sequence length, so we
#' keep only a few sizes alive: 250, 500, 1000. Counts past 1000 round up
#' to the next 250 (rare; the default token cap is 1000).
#'
#' @param n Speech-token count
#' @return Bucketed token count (250, 500, 1000, or a 250-multiple)
#' @noRd
.token_bucket <- function(n) {
    if (n <= 250L) {
        250L
    } else if (n <= 500L) {
        500L
    } else if (n <= 1000L) {
        1000L
    } else {
        as.integer(ceiling(n / 250) * 250)
    }
}

# Total VRAM (GB) for a cuda device, via nvidia-smi, memoized (it never
# changes). NA for cpu/unknown. torch exposes no exported total-memory
# query in this build, so we shell out.
.gpu_gb_cache <- new.env(parent = emptyenv())
.gpu_total_gb <- function(device) {
    if (!grepl("^cuda", device)) {
        return(NA_real_)
    }
    if (!is.null(.gpu_gb_cache[[device]])) {
        return(.gpu_gb_cache[[device]])
    }
    idx <- sub("^cuda:?", "", device)
    if (nzchar(idx)) {
        idx <- suppressWarnings(as.integer(idx))
    } else {
        idx <- 0L
    }
    if (is.na(idx)) {
        idx <- 0L
    }
    out <- tryCatch(system2("nvidia-smi",
                            c("--query-gpu=memory.total", "--format=csv,noheader,nounits",
                              paste0("--id=", idx)),
                            stdout = TRUE, stderr = FALSE),
                    error = function(e) character(0))
    gb <- NA_real_
    if (length(out) >= 1L) {
        v <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", out[1])))
        if (!is.na(v)) {
            gb <- v / 1024
        }
    }
    .gpu_gb_cache[[device]] <- gb
    gb
}

#' Per-card cap on batched-CFM batch size
#'
#' The batched (eager) CFM solve uses memory ~ batch_size * cfm_len, where
#' cfm_len = 640 + 2 * tokens (the padded mel sequence). We turn a card's
#' measured single-shot ceiling into a budget in cfm_len units and divide.
#' Known limits: the 6 GB GTX 1660 Ti does one ~1000-token solve
#' (cfm_len ~2688) before fragmentation-OOM; a 16 GB card has ~6x that
#' activation headroom. Conservative (no sweep): err toward smaller
#' batches.
#'
#' @param bucket Token bucket (250/500/1000)
#' @param device Device string ("cuda", "cuda:0", "cpu", ...)
#' @return Maximum chunks to batch together at this bucket (>= 1)
#' @noRd
.cfm_max_batch <- function(bucket, device) {
    gb <- .gpu_total_gb(device)
    budget <- if (is.na(gb)) {
        2688L
    } else if (gb < 8) {
        2688L
    } else if (gb < 24) {
        15000L
    } else {
        30000L
    }
    max(1L, budget %/% (640L + 2L * bucket))
}

#' Split text into TTS-sized chunks
#'
#' Sentences first; sentences longer than chunk_size chars are packed at
#' comma boundaries, and any clause still over chunk_size is word-split as
#' a last resort. Splitting mid-clause hurts prosody, but a long
#' comma-less run would otherwise trip the T3 text-token guard, so every
#' chunk is kept at or under chunk_size (bar a single oversize word).
#'
#' @param text Input text
#' @param chunk_size Target maximum chunk length in characters
#' @return Character vector of chunks
#' @noRd
# Emit every chunk that is contiguous from `from`, in ORIGINAL order, and
# return the new watermark.
#
# The batched path fills `audio_chunks` out of order: it walks token-length
# buckets, so a short chunk late in the text can finish in the first group.
# Streaming in completion order would deliver a player its audio scrambled,
# which is worse than not streaming, so a chunk waits until everything ahead
# of it exists.
#
# Worst case -- chunk 1 completes last -- this emits nothing until the end,
# which is exactly the non-streaming behaviour it replaces. It never emits
# less correctly, only less eagerly.
#
# The watermark is the caller's, not this function's, so calling again with
# nothing new added is silent rather than a replay of the prefix.
.emit_ready <- function(audio_chunks, from, on_chunk, total) {
    i <- from
    n <- length(audio_chunks)
    while (i <= n && !is.null(audio_chunks[[i]])) {
        on_chunk(audio_chunks[[i]], i, total)
        i <- i + 1L
    }
    i
}

.split_text_chunks <- function(text, chunk_size = 200L) {
    # Greedily pack space-joined units into pieces of <= chunk_size chars.
    pack <- function(units) {
        out <- character(0)
        cur <- ""
        for (u in units) {
            if (nzchar(cur) && nchar(cur) + 1L + nchar(u) > chunk_size) {
                out <- c(out, cur)
                cur <- u
            } else {
                if (nzchar(cur)) {
                    cur <- paste(cur, u)
                } else {
                    cur <- u
                }
            }
        }
        if (nzchar(cur)) {
            out <- c(out, cur)
        }
        out
    }
    sentences <- strsplit(text, "(?<=[.!?])\\s+", perl = TRUE)[[1]]
    chunks <- character(0)
    for (s in sentences) {
        if (nchar(s) <= chunk_size) {
            chunks <- c(chunks, s)
            next
        }
        # Oversized sentence: pack at comma boundaries, then word-split any
        # clause still over chunk_size. The word split is a last resort but
        # it keeps a long comma-less run from tripping the T3 text-token
        # guard inside generate() for serve traffic.
        for (p in pack(strsplit(s, "(?<=,)\\s+", perl = TRUE)[[1]])) {
            if (nchar(p) <= chunk_size) {
                chunks <- c(chunks, p)
            } else {
                chunks <- c(chunks, pack(strsplit(p, "\\s+", perl = TRUE)[[1]]))
            }
        }
    }
    chunks[nzchar(trimws(chunks))]
}

#' Generate speech for long text (the long-form policy layer)
#'
#' Splits at sentence boundaries (oversized sentences subdivided at commas,
#' then word-split as a last resort), resolves the voice once, and runs T3
#' on every chunk first so batching uses ACTUAL speech-token lengths rather
#' than a character estimate. Chunks are then bucketed by their real length
#' and synthesized within a per-card batch cap (sized from VRAM): a group
#' of one takes the fast traced-CFM path, a group of several runs as one
#' eager batched S3Gen solve. Audio is stitched in original order; garbage
#' is collected at each batch boundary (see
#' \code{\link{chatterbox_gc_options}}). Turbo has no batched path and is
#' synthesized serially.
#'
#' @param model Chatterbox model
#' @param text Text to synthesize
#' @param voice Voice embedding or path to reference audio (resolved once)
#' @param chunk_size Maximum characters per chunk (default 200)
#' @param max_batch Maximum chunks per batched solve. Default NULL: sized
#'   per card from VRAM. Set an integer to override.
#' @param on_chunk Optional function of \code{(audio, index, total)} called
#'   as each chunk becomes available, so a caller can start playing before
#'   the whole utterance is synthesized. \strong{Always called in original
#'   chunk order.} The default \code{NULL} accumulates as before and the
#'   return value is unchanged either way, so a caller that ignores this
#'   sees identical behaviour.
#'
#'   How much it actually streams depends on which synthesis strategy runs,
#'   and the difference is worth knowing before designing around it.
#'   \strong{Serial synthesis streams; batched synthesis does not.} Serial
#'   finishes one chunk at a time in order, so each is emitted the moment it
#'   is done. Batched runs T3 over \emph{every} chunk before any S3Gen work
#'   starts, then walks token-length buckets rather than text order -- so
#'   nothing is emitted until T3 finishes across the whole input, and a
#'   late-arriving early chunk holds back the ones behind it.
#'
#'   Today the strategy is not selectable: it follows the loaded model,
#'   because the turbo weights have no batched implementation. So the turbo
#'   model synthesizes serially and streams, and the standard model batches
#'   and does not. That coupling is incidental rather than intended --
#'   \code{\link{generate}} handles both models, so the standard weights
#'   could synthesize serially too, trading throughput for time-to-first-
#'   audio. Nothing exposes that choice yet.
#' @param ... Synthesis arguments forwarded to the T3 and S3Gen stages, as
#'   in \code{\link{generate}} (exaggeration, cfg_weight, temperature,
#'   backend, traced, normalize_text, max_new_tokens, ...)
#' @return List with audio and sample_rate
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' res <- tts_chunked(model, long_text, "reference_voice.wav")
#' write_audio(res$audio, res$sample_rate, "long.wav")
#' }
#' @export
tts_chunked <- function(model, text, voice, chunk_size = 200,
                        max_batch = NULL, on_chunk = NULL, ...) {
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }
    if (!is.null(on_chunk) && !is.function(on_chunk)) {
        stop("on_chunk must be a function of (audio, index, total), or NULL",
             call. = FALSE)
    }

    chunks <- .split_text_chunks(text, chunk_size)
    if (length(chunks) == 0L) {
        return(list(audio = numeric(0), sample_rate = S3GEN_SR))
    }

    # Resolve the voice once; re-encoding it per chunk is pure waste
    if (is.character(voice)) {
        voice <- create_voice_embedding(model, voice)
    }

    args <- list(...)
    arg_or <- function(name, default) args[[name]] %||% default
    traced <- isTRUE(arg_or("traced", FALSE))
    audio_chunks <- vector("list", length(chunks))

    # The turbo weights have no batched synthesis implementation, so this
    # path is serial. Serial is also the strategy that streams -- each chunk
    # is finished and in text order, so it goes out immediately. The two
    # facts are independent and only coincide here: `generate()` handles
    # both models, so the standard weights could run this loop too and trade
    # throughput for time-to-first-audio. Nothing selects that yet.
    if (isTRUE(model$turbo)) {
        for (i in seq_along(chunks)) {
            message(sprintf("Processing chunk %d/%d", i, length(chunks)))
            audio_chunks[[i]] <- generate(model, chunks[i], voice, ...)$audio
            if (!is.null(on_chunk)) {
                on_chunk(audio_chunks[[i]], i, length(chunks))
            }
            gc(verbose = FALSE)
        }
        return(list(audio = unlist(audio_chunks, use.names = FALSE),
                    sample_rate = S3GEN_SR))
    }

    # Run T3 on every chunk first, so batching and the memory cap use
    # ACTUAL speech-token lengths, not a char estimate (corteza review #2).
    tk <- .texts_to_speech_tokens(model, chunks, voice,
                                  normalize_text = isTRUE(arg_or("normalize_text", FALSE)),
                                  use_autocast = isTRUE(arg_or("autocast", FALSE)) &&
                                  grepl("^cuda", model$device),
                                  exaggeration = arg_or("exaggeration", 0.5),
                                  cfg_weight = arg_or("cfg_weight", 0.5),
                                  temperature = arg_or("temperature", 0.8),
                                  top_p = arg_or("top_p", 1.0), min_p = arg_or("min_p", 0.05),
                                  traced = traced, backend = arg_or("backend", "r"),
                                  top_k = arg_or("top_k", 1000L),
                                  repetition_penalty = arg_or("repetition_penalty", 1.2),
                                  max_new_tokens = arg_or("max_new_tokens", 1000L),
                                  max_cache_len = arg_or("max_cache_len", NULL))

    # Bucket by actual token length, then synthesize within the per-card
    # cap, preserving original order. A group of one takes the traced
    # single path; several run as one eager batched S3Gen solve.
    lens <- vapply(tk$token_vecs, length, integer(1L))
    buckets <- vapply(pmax(lens, 1L), .token_bucket, integer(1L))
    done <- 0L
    ## How far the in-order stream has got. This walks buckets rather than
    ## chunks, so a group can complete chunk 5 before chunk 1 exists --
    ## emitting in completion order would hand a player its audio scrambled.
    emitted <- 1L
    for (b in sort(unique(buckets))) {
        idx <- which(buckets == b)
        cap <- max(1L, max_batch %||% .cfm_max_batch(b, model$device))
        for (grp in split(idx, ceiling(seq_along(idx) / cap))) {
            res <- .s3gen_batch_from_tokens(model, tk$token_vecs[grp],
                tk$eos[grp], voice, traced = traced)
            for (j in seq_along(grp)) {
                audio_chunks[[grp[j]]] <- res[[j]]$audio
            }
            if (!is.null(on_chunk)) {
                emitted <- .emit_ready(audio_chunks, emitted, on_chunk,
                                       length(chunks))
            }
            done <- done + length(grp)
            message(sprintf("Synthesized %d/%d chunks", done, length(chunks)))
            gc(verbose = FALSE)
        }
    }

    list(audio = unlist(audio_chunks, use.names = FALSE),
         sample_rate = S3GEN_SR)
}

# ============================================================================
# Print Methods
# ============================================================================

#' Print method for chatterbox
#'
#' @param x Chatterbox model
#' @param ... Ignored
#' @return \code{x}, invisibly. Called for the side effect of printing a
#'   summary of the model to the console.
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
#' @return \code{x}, invisibly. Called for the side effect of printing the
#'   embedding's shape and sample rate to the console.
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
#' @return The \code{\link{generate}} result list (audio, sample_rate,
#'   ...). When \code{output_path} is set the audio is also written there
#'   (the list gains a \code{path} element) and the list is returned
#'   invisibly so the audio vector does not print.
#' @examples
#' \dontrun{
#' quick_tts("Hello!", "reference_voice.wav", "out.wav")
#' }
#' @export
quick_tts <- function(text, reference_audio, output_path = NULL,
                      device = "cpu", autocast = NULL, turbo = FALSE) {
    # Create and load model in one call
    model <- chatterbox(device, turbo = turbo)

    res <- generate(model, text, reference_audio, autocast = autocast,
                    output_path = output_path)
    if (is.null(output_path)) {
        res
    } else {
        invisible(res)
    }
}
