% tinyrox says don't edit this manually, but it can't stop you!
\name{serve_chatterbox}
\alias{serve_chatterbox}
\title{Serve Chatterbox as HTTP Microservice}
\usage{
serve_chatterbox(port = 7810L, host = "0.0.0.0", device = "cuda", traced = TRUE)
}
\arguments{
\item{port}{Integer. Port to listen on. Default 7810.}

\item{host}{Character. Bind address. Default "0.0.0.0".}

\item{device}{Device to use ("cpu" or "cuda"). Default "cuda".}

\item{traced}{Use JIT-traced inference. Default TRUE.}
}
\value{
Does not return (runs until interrupted).
}
\description{
Convenience wrapper that starts a Chatterbox TTS server using
\code{gpu.ctl::serve()}. The model is loaded once at startup and
reused for all requests.
}
\details{
Exposes the following endpoints:
\describe{
  \item{GET /health}{Returns JSON with model status}
  \item{POST /audio/speech}{Generate speech. JSON body with fields:
    \code{input} (text), \code{voice} (path to reference audio),
    \code{exaggeration} (0-1), \code{cfg_weight} (0-1),
    \code{temperature} (0-5). Returns raw WAV audio.}
}

Requires the gpu.ctl package. Install with:
\code{install.packages("gpu.ctl")}
}
\examples{
\dontrun{
serve_chatterbox()
# curl -X POST http://localhost:7810/audio/speech \
#   -H "Content-Type: application/json" \
#   -d '{"input":"Hello world","voice":"/path/to/ref.wav"}' \
#   --output hello.wav
}
}
