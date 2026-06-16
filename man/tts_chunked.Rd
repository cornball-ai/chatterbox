% tinyrox says don't edit this manually, but it can't stop you!
\name{tts_chunked}
\alias{tts_chunked}
\title{Generate speech for long text (the long-form policy layer)}
\usage{
tts_chunked(model, text, voice, chunk_size = 200, max_batch = NULL, ...)
}
\arguments{
\item{model}{Chatterbox model}

\item{text}{Text to synthesize}

\item{voice}{Voice embedding or path to reference audio (resolved once)}

\item{chunk_size}{Maximum characters per chunk (default 200)}

\item{max_batch}{Maximum chunks per batched solve. Default NULL: sized
per card from VRAM. Set an integer to override.}

\item{...}{Synthesis arguments forwarded to the T3 and S3Gen stages, as
in \code{\link{generate}} (exaggeration, cfg_weight, temperature,
backend, traced, normalize_text, max_new_tokens, ...)}
}
\value{
List with audio and sample_rate
}
\description{
Splits at sentence boundaries (oversized sentences subdivided at commas,
then word-split as a last resort), resolves the voice once, and runs T3
on every chunk first so batching uses ACTUAL speech-token lengths rather
than a character estimate. Chunks are then bucketed by their real length
and synthesized within a per-card batch cap (sized from VRAM): a group
of one takes the fast traced-CFM path, a group of several runs as one
eager batched S3Gen solve. Audio is stitched in original order; garbage
is collected at each batch boundary (see
\code{\link{chatterbox_gc_options}}). Turbo has no batched path and is
synthesized serially.
}
