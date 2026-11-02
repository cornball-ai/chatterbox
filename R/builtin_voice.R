# The built-in voice: Python chatterbox's default conditionals, converted.
#
# Python speaks in a default voice when no reference clip is given: the
# conditionals its prepare_conditionals() computed once from a reference
# clip and saved beside the weights as conds.pt, a nested torch pickle R
# torch cannot read. inst/voices/ holds the same tensors rewritten as
# safetensors, one file per model set (inst/voices/README.md records the
# conversion and the digests), so generate() can speak with no reference
# clip, as Python does.

.builtin_voice_cache <- new.env(parent = emptyenv())

#' The built-in voice
#'
#' Python chatterbox falls back to a default voice when no reference audio
#' is given: conditionals computed once from a reference clip and shipped
#' with the weights as \code{conds.pt}. This package ships the same
#' conditionals, rewritten as safetensors under \code{inst/voices}: one
#' set for the standard model and one for turbo, which differ (turbo's
#' conditioning prompt is 375 speech tokens to the standard model's 150).
#' \code{\link{generate}} and the functions built on it use this voice
#' when \code{voice} is \code{NULL}.
#'
#' @param turbo Logical. The turbo model's voice (\code{TRUE}) or the
#'   standard model's (\code{FALSE}, the default). A model object may be
#'   given instead, and its variant and device are used.
#' @param device Device to hold the tensors on. Ignored when a model is
#'   given.
#' @return A \code{voice_embedding}, as \code{\link{create_voice_embedding}}
#'   returns. Its \code{sample_rate} is \code{NA}: there is no reference
#'   clip behind it. Loaded once per model set and device.
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' res <- generate(model, "No reference clip was given.")
#' dim(builtin_voice(model)$cond_prompt_speech_tokens)
#' }
#' @export
builtin_voice <- function(turbo = FALSE, device = "cpu") {
    if (inherits(turbo, "chatterbox")) {
        device <- turbo$device
        turbo <- isTRUE(turbo$turbo)
    }
    name <- if (isTRUE(turbo)) "chatterbox-turbo" else "chatterbox"
    key <- paste(name, device)
    voice <- .builtin_voice_cache[[key]]
    if (!is.null(voice)) {
        return(voice)
    }
    path <- system.file("voices", paste0(name, ".safetensors"),
                        package = "chatterbox")
    if (!nzchar(path)) {
        stop("the built-in voice for ", name, " is missing from the ",
             "installed package", call. = FALSE)
    }
    st <- read_safetensors(path, device = device)
    voice <- structure(
                       list(
                            ve_embedding = st[["t3.speaker_emb"]],
                            cond_prompt_speech_tokens = st[["t3.cond_prompt_speech_tokens"]],
                            ref_dict = list(
                prompt_token = st[["gen.prompt_token"]],
                prompt_token_len = st[["gen.prompt_token_len"]],
                prompt_feat = st[["gen.prompt_feat"]],
                prompt_feat_len = NULL,
                embedding = st[["gen.embedding"]]
            ),
                            sample_rate = NA_integer_
        ),
                       class = "voice_embedding"
    )
    .builtin_voice_cache[[key]] <- voice
    voice
}

# A voice as generate() takes it: a voice_embedding as given, a path
# encoded now, or NULL for the built-in voice of this model's variant.
.resolve_voice <- function(model, voice, use_autocast = FALSE) {
    if (is.null(voice)) {
        return(builtin_voice(model))
    }
    if (is.character(voice)) {
        return(create_voice_embedding(model, voice, autocast = use_autocast))
    }
    if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object, a path to reference ",
             "audio, or NULL for the built-in voice", call. = FALSE)
    }
    voice
}
