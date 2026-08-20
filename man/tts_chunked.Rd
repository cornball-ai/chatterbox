% tinyrox says don't edit this manually, but it can't stop you!
\name{tts_chunked}
\alias{tts_chunked}
\title{Generate speech for long text (the long-form policy layer)}
\usage{
tts_chunked(
  model,
  text,
  voice,
  chunk_size = 200,
  max_batch = NULL,
  on_chunk = NULL,
  ...
)
}
\arguments{
\item{model}{Chatterbox model}

\item{text}{Text to synthesize}

\item{voice}{Voice embedding or path to reference audio (resolved once)}

\item{chunk_size}{Maximum characters per chunk (default 200)}

\item{max_batch}{Maximum chunks per batched solve. Default NULL: sized
per card from VRAM. Set an integer to override.}

\item{on_chunk}{Optional function of \code{(audio, index, total)} called
as each chunk becomes available, so a caller can start playing before
the whole utterance is synthesized. \strong{Always called in original
chunk order.} The default \code{NULL} accumulates as before and the
return value is unchanged either way, so a caller that ignores this
sees identical behaviour.

How much it actually streams depends on which synthesis strategy runs,
and the difference is worth knowing before designing around it.
\strong{Serial synthesis streams; batched synthesis does not.} Serial
finishes one chunk at a time in order, so each is emitted the moment it
is done. Batched runs T3 over \emph{every} chunk before any S3Gen work
starts, then walks token-length buckets rather than text order -- so
nothing is emitted until T3 finishes across the whole input, and a
late-arriving early chunk holds back the ones behind it.

Today the strategy is not selectable: it follows the loaded model,
because the turbo weights have no batched implementation. So the turbo
model synthesizes serially and streams, and the standard model batches
and does not. That coupling is incidental rather than intended --
\code{\link{generate}} handles both models, so the standard weights
could synthesize serially too, trading throughput for time-to-first-
audio. Nothing exposes that choice yet.}

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
\examples{
\dontrun{
model <- chatterbox("cuda")
res <- tts_chunked(model, long_text, "reference_voice.wav")
write_audio(res$audio, res$sample_rate, "long.wav")
}
}
