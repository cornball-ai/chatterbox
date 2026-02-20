# chatterbox HTTP handler for gpu.ctl::serve()

#' Create Chatterbox HTTP Request Handler
#'
#' Returns a handler function suitable for \code{gpu.ctl::serve()}.
#' The model is loaded once at handler creation and reused for all requests.
#'
#' @param device Device to use ("cpu" or "cuda"). Default "cuda".
#' @param traced Use JIT-traced inference for faster generation. Default TRUE.
#'
#' @return A function with signature
#'   \code{function(method, path, headers, body)} for use with
#'   \code{gpu.ctl::serve()}.
#' @export
#'
#' @examples
#' \dontrun{
#' handler <- chatterbox_handler()
#' gpu.ctl::serve(7810, handler)
#' }
chatterbox_handler <- function (device = "cuda", traced = TRUE, turbo = FALSE)
{
    variant <- if (turbo) "Turbo" else "Standard"
    message("Loading Chatterbox ", variant, " model...")
    model <- chatterbox(device, turbo = turbo)
    model <- load_chatterbox(model)

    # Pre-create voice embedding cache
    voice_cache <- new.env(parent = emptyenv())

    function (method, path, headers, body) {
        if (path == "/health") {
            return(list(status = "ok", model = "chatterbox", device = device))
        }

        if (path == "/audio/speech" && method == "POST") {
            req <- jsonlite::fromJSON(body)

            text <- req$input
            if (is.null(text) || !nzchar(text)) {
                return(list(status = 400L, body = '{"error":"input is required"}'))
            }

            # Voice: file path to reference audio
            voice_path <- req$voice
            if (is.null(voice_path) || !nzchar(voice_path)) {
                return(list(
                    status = 400L,
                    body = '{"error":"voice (path to reference audio) is required"}'
                ))
            }

            if (!file.exists(voice_path)) {
                return(list(
                    status = 400L,
                    body = paste0('{"error":"voice file not found: ', voice_path, '"}')
                ))
            }

            # Cache voice embeddings by file path
            ve <- voice_cache[[voice_path]]
            if (is.null(ve)) {
                ve <- create_voice_embedding(model, voice_path)
                voice_cache[[voice_path]] <- ve
            }

            # Generation parameters (match Docker API defaults)
            exaggeration <- if (!is.null(req$exaggeration)) req$exaggeration else 0.5
            cfg_weight <- if (!is.null(req$cfg_weight)) req$cfg_weight else 0.5
            temperature <- if (!is.null(req$temperature)) req$temperature else 0.8

            # Generate audio
            result <- generate(
                model, text, ve,
                exaggeration = exaggeration,
                cfg_weight = cfg_weight,
                temperature = temperature,
                traced = traced
            )

            # Write WAV to temp file, read back as raw bytes
            tmp <- tempfile(fileext = ".wav")
            on.exit(unlink(tmp), add = TRUE)
            write_audio(result$audio, result$sample_rate, tmp)
            wav_bytes <- readBin(tmp, "raw", file.info(tmp)$size)

            return(list(
                status = 200L,
                headers = list("content-type" = "audio/wav"),
                body = wav_bytes
            ))
        }

        list(status = 404L, body = '{"error":"not found"}')
    }
}

#' Serve Chatterbox as HTTP Microservice
#'
#' Convenience wrapper that starts a Chatterbox TTS server using
#' \code{gpu.ctl::serve()}. The model is loaded once at startup and
#' reused for all requests.
#'
#' @param port Integer. Port to listen on. Default 7810.
#' @param host Character. Bind address. Default "0.0.0.0".
#' @param device Device to use ("cpu" or "cuda"). Default "cuda".
#' @param traced Use JIT-traced inference. Default TRUE.
#'
#' @details
#' Exposes the following endpoints:
#' \describe{
#'   \item{GET /health}{Returns JSON with model status}
#'   \item{POST /audio/speech}{Generate speech. JSON body with fields:
#'     \code{input} (text), \code{voice} (path to reference audio),
#'     \code{exaggeration} (0-1), \code{cfg_weight} (0-1),
#'     \code{temperature} (0-5). Returns raw WAV audio.}
#' }
#'
#' Requires the gpu.ctl package. Install with:
#' \code{install.packages("gpu.ctl")}
#'
#' @return Does not return (runs until interrupted).
#' @export
#'
#' @examples
#' \dontrun{
#' serve_chatterbox()
#' # curl -X POST http://localhost:7810/audio/speech \
#' #   -H "Content-Type: application/json" \
#' #   -d '{"input":"Hello world","voice":"/path/to/ref.wav"}' \
#' #   --output hello.wav
#' }
serve_chatterbox <- function (port = 7810L, host = "0.0.0.0",
                              device = "cuda", traced = TRUE, turbo = FALSE)
{
    if (!requireNamespace("gpu.ctl", quietly = TRUE)) {
        stop(
            "gpu.ctl is required for serve_chatterbox(). ",
            "Install with: install.packages('gpu.ctl')",
            call. = FALSE
        )
    }
    gpu.ctl::serve(port, chatterbox_handler(device = device, traced = traced,
                                             turbo = turbo),
                   host = host)
}
