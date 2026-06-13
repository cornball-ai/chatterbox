% tinyrox says don't edit this manually, but it can't stop you!
\name{serve}
\alias{serve}
\title{Serve chatterbox over HTTP}
\usage{
serve(port = 7810L, device = "cuda", voices_dir = NULL, turbo = FALSE,
      timeout = 300L, max_body = 10L * 1024L^2, warmup = TRUE)
}
\arguments{
\item{port}{Integer. TCP port to listen on. Default 7810.}

\item{device}{Character. Torch device for the model ("cuda", "cpu", "mps").}

\item{voices_dir}{Character. Directory of voice reference files. Defaults to
the \code{TTS_VOICES_DIR} env var, then \code{~/.cornball/voices}.}

\item{turbo}{Logical. Serve the Chatterbox Turbo model.}

\item{timeout}{Integer. Per-connection I/O timeout in seconds (guards against
stalled clients). Default 300.}

\item{max_body}{Integer. Maximum request body size in bytes. Default 10 MB.}

\item{warmup}{Logical. Run one short synthesis at startup to trigger the
one-time JIT tracing, so the first client request isn't slow. Default TRUE.}
}
\value{
Does not return normally; runs until interrupted.
}
\description{
Starts a blocking HTTP server that loads the chatterbox model once and answers
OpenAI-compatible TTS requests. Intended as a drop-in replacement for the
chatterbox TTS container: point an HTTP client (e.g. \pkg{tts.api}) at
\code{http://<host>:<port>} and it serves the same endpoints.
}
\details{
Endpoints:
\itemize{
\item \code{GET /health} - liveness probe, returns \code{{"status":"ok"}}.
\item \code{GET /v1/audio/voices} - lists voice names in \code{voices_dir}.
\item \code{POST /v1/audio/speech} - body \code{{input, voice,
response_format, exaggeration, cfg_weight, temperature}}; returns the
synthesized audio bytes. \code{voice} is a voice-library name (resolved
against \code{voices_dir}) or a path to a reference audio file.
}

The server is single-threaded and runs until interrupted. Run it under a
process supervisor (systemd, a container CMD, tmux) for persistence. An
example systemd unit ships with the package:
\code{system.file("chatterbox.service", package = "chatterbox")}.

}
