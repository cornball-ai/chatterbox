% tinyrox says don't edit this manually, but it can't stop you!
\name{tts_chunked}
\alias{tts_chunked}
\title{Generate speech in chunks (for long texts)}
\usage{
tts_chunked(model, text, voice, chunk_size = 200, ...)
}
\arguments{
\item{model}{Chatterbox model}

\item{text}{Text to synthesize}

\item{voice}{Voice embedding}

\item{chunk_size}{Maximum characters per chunk (default 200)}

\item{...}{Additional arguments passed to generate()}
}
\value{
List with audio and sample_rate
}
\description{
Splits at sentence boundaries; sentences longer than chunk_size
characters are further split at commas. Collects garbage once per
chunk (see \code{\link{chatterbox_gc_options}}).
}
