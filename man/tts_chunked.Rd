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
  strategy = c("auto", "serial", "batched"),
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

\item{on_chunk}{Optional function of \code{(audio, index, total, text)}
called as each chunk becomes available, so a caller can start playing
before the whole utterance is synthesized. \strong{Always called in
original chunk order.} The default \code{NULL} accumulates as before and
the return value is unchanged either way, so a caller that ignores this
sees identical behaviour.

\code{text} is the piece this chunk says. It is handed over at emit
time rather than left to be recovered afterwards because a caller that
has to report what was heard needs the split that was actually spoken,
and because synthesis may be abandoned part-way -- on a barge-in, say --
in which case the return value never arrives and the emitted pieces are
the only record there is.

How much it actually streams depends on \code{strategy}:
\strong{serial streams, batched does not.} See there.}

\item{strategy}{Synthesis strategy.
\describe{
\item{\code{"auto"}}{(default) Batched where the weights support it,
serial where they do not. This is what every existing caller
already gets.}
\item{\code{"serial"}}{One chunk at a time in text order. Each is
finished when it is done, so \code{on_chunk} fires immediately and
audio can start playing after chunk 1. Costs throughput: one CFM
solve per chunk instead of one per batch. Works on both models.}
\item{\code{"batched"}}{Runs T3 over \emph{every} chunk, buckets them
by measured speech-token length, and synthesizes each bucket as one
padded S3Gen solve. Better throughput. Nothing can be emitted until
T3 finishes across the whole input, and buckets do not follow text
order, so \code{on_chunk} fires late and in held-back runs.
\strong{Standard weights only} -- there is no batched implementation
for turbo, and asking for it there is an error rather than a silent
fall back to serial.}
}
For a conversational turn, time-to-first-audio is the thing that
matters and throughput is nearly irrelevant, so \code{"serial"} is the
one to want. For synthesizing a long document, batched wins outright.}

\item{...}{Synthesis arguments forwarded to the T3 and S3Gen stages, as
in \code{\link{generate}} (exaggeration, cfg_weight, temperature,
backend, traced, normalize_text, max_new_tokens, ...)}
}
\value{
List with \code{audio}, \code{sample_rate}, and \code{chunks} --
  the text pieces that were synthesized, in spoken order, one per chunk.
  \code{chunks[[i]]} is what chunk \code{i} says, which is the mapping any
  report of how far playback got depends on. Present whether or not
  \code{on_chunk} was supplied.
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
