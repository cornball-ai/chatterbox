# Voice conversion (port of Python chatterbox's vc.py).
# Speech-to-speech: re-synthesize source speech in the target voice.
# Skips T3 entirely - the source audio is tokenized by the S3 tokenizer
# and S3Gen regenerates it conditioned on the target speaker.

#' Convert speech to a target voice
#'
#' Re-synthesizes \code{audio} so the same words and prosody come out in
#' the target voice (Python chatterbox's \code{ChatterboxVC}). No text
#' or T3 generation is involved: the source speech is tokenized directly
#' (25 tokens/s) and S3Gen renders the tokens with the target speaker's
#' conditioning, so the result follows the source's timing.
#'
#' @param model Loaded chatterbox model (standard, not turbo)
#' @param audio Source speech (file path, numeric vector, or torch
#'   tensor)
#' @param voice Target voice: a voice_embedding from
#'   \code{\link{create_voice_embedding}} (or
#'   \code{\link{load_voice_embedding}}), or a path to reference audio
#' @param sample_rate Sample rate of \code{audio} (if not a file)
#' @return List with \code{audio} (numeric vector), \code{sample_rate}
#'   (24000), and \code{audio_sec}, like \code{\link{generate}}
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' res <- voice_convert(model, "source_speech.wav", "target_voice.wav")
#' write_audio(res$audio, res$sample_rate, "converted.wav")
#' }
#' @export
voice_convert <- function(model, audio, voice, sample_rate = NULL) {
    if (!is_loaded(model)) {
        stop("Model not loaded. Call load_chatterbox() first.")
    }
    if (isTRUE(model$turbo)) {
        stop("Voice conversion uses the standard S3Gen decoder; ",
             "load a standard (non-turbo) model.")
    }

    # Target voice conditioning (only ref_dict is used; Python VC's
    # embed_ref caps the reference at 10 s, as create_voice_embedding
    # already does)
    if (is.character(voice)) {
        voice <- create_voice_embedding(model, voice)
    }
    if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object or path to ",
             "reference audio")
    }

    # Source speech at 16 kHz for the S3 tokenizer
    if (is.character(audio)) {
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
    if (sample_rate != S3_SR) {
        samples <- resample_audio(samples, sample_rate, S3_SR)
    }

    device <- model$device
    audio_16k <- torch::torch_tensor(samples,
                                     dtype = torch::torch_float32())$unsqueeze(1)$to(device = device)

    torch::with_no_grad({
        # Full-length tokenization: VC keeps the source's timing
        tok <- model$s3gen$tokenizer$forward(audio_16k)
        result <- model$s3gen$inference(
                                        speech_tokens = tok$tokens$to(device = device),
                                        ref_dict = voice$ref_dict,
                                        finalize = TRUE
        )
    })

    audio_samples <- as.numeric(result[[1]]$squeeze()$cpu())
    list(
         audio = audio_samples,
         sample_rate = S3GEN_SR,
         audio_sec = length(audio_samples) / S3GEN_SR
    )
}
