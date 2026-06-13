# serve.R
# Minimal HTTP server exposing the chatterbox package over an OpenAI-compatible
# /v1/audio/speech endpoint. Built on base R sockets (serverSocket/socketAccept)
# so it adds no dependencies and runs as a single persistent process: the model
# loads once and stays resident on the GPU (no fork, so the CUDA context is never
# invalidated). Requests are served one at a time, which is the natural fit for a
# single-GPU TTS box.

#' Serve chatterbox over HTTP
#'
#' Starts a blocking HTTP server that loads the chatterbox model once and answers
#' OpenAI-compatible TTS requests. Intended as a drop-in replacement for the
#' chatterbox TTS container: point an HTTP client (e.g. \pkg{tts.api}) at
#' \code{http://<host>:<port>} and it serves the same endpoints.
#'
#' Endpoints:
#' \itemize{
#'   \item \code{GET /health} - liveness probe, returns \code{{"status":"ok"}}.
#'   \item \code{GET /v1/audio/voices} - lists voice names in \code{voices_dir}.
#'   \item \code{POST /v1/audio/speech} - body \code{{input, voice,
#'     response_format, exaggeration, cfg_weight, temperature}}; returns the
#'     synthesized audio bytes. \code{voice} is a voice-library name (resolved
#'     against \code{voices_dir}) or a path to a reference audio file.
#' }
#'
#' The server is single-threaded and runs until interrupted. Run it under a
#' process supervisor (systemd, a container CMD, tmux) for persistence.
#'
#' @param port Integer. TCP port to listen on. Default 7810.
#' @param device Character. Torch device for the model ("cuda", "cpu", "mps").
#' @param voices_dir Character. Directory of voice reference files. Defaults to
#'   the \code{TTS_VOICES_DIR} env var, then \code{~/.cornball/voices}.
#' @param turbo Logical. Serve the Chatterbox Turbo model.
#' @param timeout Integer. Per-connection I/O timeout in seconds (guards against
#'   stalled clients). Default 300.
#' @param max_body Integer. Maximum request body size in bytes. Default 10 MB.
#' @param warmup Logical. Run one short synthesis at startup to trigger the
#'   one-time JIT tracing, so the first client request isn't slow. Default TRUE.
#' @return Does not return normally; runs until interrupted.
#' @export
serve <- function(port = 7810L, device = "cuda", voices_dir = NULL,
                  turbo = FALSE, timeout = 300L, max_body = 10L * 1024L^2,
                  warmup = TRUE) {
    if (is.null(voices_dir)) {
        voices_dir <- Sys.getenv("TTS_VOICES_DIR", "~/.cornball/voices")
    }
    voices_dir <- path.expand(voices_dir)
    if (!dir.exists(voices_dir)) {
        warning("voices_dir does not exist: ", voices_dir)
    }

    message("Loading chatterbox", if (turbo) " turbo" else "", " model on ",
            device, " ...")
    model <- chatterbox(device, turbo = turbo)
    model <- if (turbo) load_chatterbox_turbo(model) else load_chatterbox(model)
    message("Model loaded. Voices dir: ", voices_dir)

    if (isTRUE(warmup)) {
        vfiles <- list.files(voices_dir, pattern = "\\.(wav|mp3|m4a|flac)$",
                             full.names = TRUE, ignore.case = TRUE)
        if (length(vfiles) > 0L) {
            message("Warming up (one-time tracing) ...")
            tryCatch(
                generate(model, "Warming up.", vfiles[1],
                         traced = !isTRUE(model$turbo)),
                error = function(e) message("warmup skipped: ",
                                            conditionMessage(e))
            )
            message("Warmup done.")
        }
    }

    srv <- serverSocket(port)
    on.exit(close(srv), add = TRUE)
    message("chatterbox::serve listening on port ", port,
            " (interrupt to stop)")

    repeat {
        con <- tryCatch(
            socketAccept(srv, blocking = TRUE, open = "r+b",
                         timeout = timeout),
            error = function(e) {
                message("accept error: ", conditionMessage(e))
                Sys.sleep(0.5) # avoid a busy-spin on a bad server socket
                NULL
            }
        )
        if (is.null(con)) next
        tryCatch({
            req <- .serve_read_request(con, max_body)
            if (!is.null(req)) {
                resp <- tryCatch(
                    .serve_route(req, model, voices_dir),
                    error = function(e) .serve_err(500L, conditionMessage(e))
                )
                .serve_send(con, resp$status, resp$content_type, resp$body)
            }
        },
        error = function(e) message("request error: ", conditionMessage(e)),
        finally = try(close(con), silent = TRUE))
    }
}

# Null/empty coalescing
.serve_or <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

# Read and parse one HTTP request from a connection. Returns a list with
# method/path/headers/body, or NULL on a closed/incomplete/oversized header.
.serve_read_request <- function(con, max_body) {
    term <- as.raw(c(13L, 10L, 13L, 10L)) # CRLFCRLF
    buf <- raw(0)
    max_header <- 65536L
    repeat {
        b <- readBin(con, "raw", n = 1L)
        if (length(b) == 0L) return(NULL) # closed or timed out mid-header
        buf <- c(buf, b)
        n <- length(buf)
        if (n >= 4L && identical(buf[(n - 3L):n], term)) break
        if (n > max_header) return(NULL)
    }

    lines <- strsplit(rawToChar(buf), "\r\n", fixed = TRUE)[[1]]
    req_line <- strsplit(lines[1], " ", fixed = TRUE)[[1]]
    if (length(req_line) < 2L) return(NULL)
    method <- req_line[1]
    path <- req_line[2]

    hdr <- list()
    if (length(lines) > 1L) {
        for (ln in lines[-1L]) {
            if (!nzchar(ln)) next
            pos <- regexpr(":", ln, fixed = TRUE)
            if (pos < 1L) next
            key <- tolower(trimws(substr(ln, 1L, pos - 1L)))
            hdr[[key]] <- trimws(substr(ln, pos + 1L, nchar(ln)))
        }
    }

    cl <- hdr[["content-length"]]
    clen <- if (is.null(cl)) 0L else suppressWarnings(as.integer(cl))
    if (length(clen) != 1L || is.na(clen) || clen < 0L) clen <- 0L
    if (clen > max_body) {
        return(list(method = method, path = path, headers = hdr,
                    body = raw(0), too_large = TRUE))
    }

    body <- if (clen > 0L) .serve_read_n(con, clen) else raw(0)
    list(method = method, path = path, headers = hdr, body = body)
}

# Read exactly n bytes (or until the stream ends).
.serve_read_n <- function(con, n) {
    out <- raw(0)
    while (length(out) < n) {
        chunk <- readBin(con, "raw", n = n - length(out))
        if (length(chunk) == 0L) break
        out <- c(out, chunk)
    }
    out
}

# Write an HTTP/1.1 response and close (Connection: close).
.serve_send <- function(con, status, content_type, body) {
    if (is.character(body)) body <- charToRaw(body)
    reason <- switch(as.character(status),
        "200" = "OK", "400" = "Bad Request", "404" = "Not Found",
        "405" = "Method Not Allowed", "413" = "Payload Too Large",
        "500" = "Internal Server Error", "Unknown")
    head <- paste0(
        sprintf("HTTP/1.1 %d %s\r\n", status, reason),
        sprintf("Content-Type: %s\r\n", content_type),
        sprintf("Content-Length: %d\r\n", length(body)),
        "Connection: close\r\n\r\n")
    writeBin(c(charToRaw(head), body), con)
    flush(con)
}

# Dispatch a parsed request to a handler.
.serve_route <- function(req, model, voices_dir) {
    if (isTRUE(req$too_large)) {
        return(.serve_err(413L, "request body too large"))
    }
    path <- sub("\\?.*$", "", req$path)

    if (identical(req$method, "GET") && path == "/health") {
        tag <- if (isTRUE(model$turbo)) "chatterbox-turbo" else "chatterbox"
        return(.serve_json(list(status = "ok", model = tag)))
    }
    if (identical(req$method, "GET") &&
        path %in% c("/v1/audio/voices", "/voices")) {
        return(.serve_json(list(voices = .serve_list_voices(voices_dir))))
    }
    if (identical(req$method, "POST") && path == "/v1/audio/speech") {
        return(.serve_speech(req, model, voices_dir))
    }
    .serve_err(404L, "not found")
}

# Synthesize speech for a /v1/audio/speech request.
.serve_speech <- function(req, model, voices_dir) {
    body <- tryCatch(jsonlite::fromJSON(rawToChar(req$body)),
                     error = function(e) NULL)
    if (is.null(body) || is.null(body$input) || is.null(body$voice)) {
        return(.serve_err(400L, "'input' and 'voice' are required"))
    }

    voice_path <- .serve_resolve_voice(body$voice, voices_dir)
    if (is.null(voice_path)) {
        return(.serve_err(400L, paste0("voice not found: ", body$voice)))
    }

    gen_args <- list(model = model, text = body$input, voice = voice_path,
                     traced = !isTRUE(model$turbo))
    if (!is.null(body$exaggeration)) gen_args$exaggeration <- body$exaggeration
    if (!is.null(body$cfg_weight)) gen_args$cfg_weight <- body$cfg_weight
    if (!is.null(body$temperature)) gen_args$temperature <- body$temperature

    res <- do.call(generate, gen_args)

    tmp_wav <- tempfile(fileext = ".wav")
    on.exit(unlink(tmp_wav), add = TRUE)
    write_audio(res$audio, res$sample_rate, tmp_wav)

    fmt <- tolower(.serve_or(body$response_format, "wav"))
    if (fmt == "wav") {
        out <- tmp_wav
    } else {
        out <- tempfile(fileext = paste0(".", fmt))
        on.exit(unlink(out), add = TRUE)
        status <- system2("ffmpeg", c("-y", "-i", shQuote(tmp_wav),
                                      shQuote(out)),
                          stdout = FALSE, stderr = FALSE)
        if (status != 0L || !file.exists(out)) {
            return(.serve_err(500L, paste0("transcode to ", fmt, " failed")))
        }
    }

    audio <- readBin(out, "raw", n = file.info(out)$size)
    list(status = 200L, content_type = .serve_ctype(fmt), body = audio)
}

# Resolve a voice name (or path) to a reference audio file.
.serve_resolve_voice <- function(voice, voices_dir) {
    if (file.exists(voice)) return(voice)
    files <- list.files(voices_dir, pattern = "\\.(wav|mp3|m4a|flac)$",
                        full.names = TRUE, ignore.case = TRUE)
    if (length(files) == 0L) return(NULL)
    names_no_ext <- tools::file_path_sans_ext(basename(files))
    idx <- match(tolower(voice), tolower(names_no_ext))
    if (is.na(idx)) return(NULL)
    files[idx]
}

# List voice-library names.
.serve_list_voices <- function(voices_dir) {
    files <- list.files(voices_dir, pattern = "\\.(wav|mp3|m4a|flac)$",
                        ignore.case = TRUE)
    tools::file_path_sans_ext(files)
}

# Content-Type for an audio format extension.
.serve_ctype <- function(fmt) {
    switch(fmt,
        mp3 = "audio/mpeg", wav = "audio/wav", ogg = "audio/ogg",
        flac = "audio/flac", opus = "audio/opus", m4a = "audio/mp4",
        "application/octet-stream")
}

# JSON 200 response.
.serve_json <- function(obj) {
    list(status = 200L, content_type = "application/json",
         body = jsonlite::toJSON(obj, auto_unbox = TRUE))
}

# JSON error response in the OpenAI {"error":{"message":...}} shape.
.serve_err <- function(status, msg) {
    list(status = status, content_type = "application/json",
         body = jsonlite::toJSON(list(error = list(message = msg)),
                                 auto_unbox = TRUE))
}
