# Voice embedding persistence (torch_save-based presets).
# R analogue of Python 0.1.7's Conditionals.save()/load(): prepare a
# voice once, reuse it across sessions without the reference audio.

# Apply f to every tensor in a nested list structure
.map_tensors <- function(x, f) {
    if (inherits(x, "torch_tensor")) {
        return(f(x))
    }
    if (is.list(x)) {
        return(lapply(x, .map_tensors, f = f))
    }
    x
}

#' Save a voice embedding to disk
#'
#' Persists a prepared voice (the R analogue of Python 0.1.7's
#' \code{Conditionals.save()}) so it can be reused across sessions
#' without the reference audio or recomputation. Tensors are moved to
#' CPU before saving; the format is \code{\link[torch]{torch_save}}
#' (not compatible with Python's \code{.pt} conditionals).
#'
#' @param voice Voice embedding from \code{\link{create_voice_embedding}}
#' @param path Output file path (suggested extension: .rds-like custom,
#'   e.g. "narrator.voice")
#' @return \code{path}, invisibly
#' @examples
#' \dontrun{
#' model <- chatterbox("cuda")
#' voice <- create_voice_embedding(model, "reference_voice.wav")
#' save_voice_embedding(voice, file.path(tempdir(), "narrator.voice"))
#' }
#' @export
save_voice_embedding <- function(voice, path) {
    if (!inherits(voice, "voice_embedding")) {
        stop("voice must be a voice_embedding object")
    }
    obj <- .map_tensors(unclass(voice), function(t) t$cpu())
    torch::torch_save(obj, path)
    invisible(path)
}

#' Load a voice embedding from disk
#'
#' @param path File written by \code{\link{save_voice_embedding}}
#' @param device Device to load tensors to (default "cpu"; use the
#'   model's device, e.g. "cuda", for generation)
#' @return A voice_embedding object
#' @examples
#' \dontrun{
#' voice <- load_voice_embedding("narrator.voice", device = "cuda")
#' model <- chatterbox("cuda")
#' res <- generate(model, "Loaded a saved voice.", voice)
#' }
#' @export
load_voice_embedding <- function(path, device = "cpu") {
    obj <- torch::torch_load(path)
    obj <- .map_tensors(obj, function(t) t$to(device = device))
    structure(obj, class = "voice_embedding")
}
