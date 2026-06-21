% tinyrox says don't edit this manually, but it can't stop you!
\name{voice_convert}
\alias{voice_convert}
\title{Convert speech to a target voice}
\usage{
voice_convert(model, audio, voice, sample_rate = NULL)
}
\arguments{
\item{model}{Loaded chatterbox model (standard, not turbo)}

\item{audio}{Source speech (file path, numeric vector, or torch
tensor)}

\item{voice}{Target voice: a voice_embedding from
\code{\link{create_voice_embedding}} (or
\code{\link{load_voice_embedding}}), or a path to reference audio}

\item{sample_rate}{Sample rate of \code{audio} (if not a file)}
}
\value{
List with \code{audio} (numeric vector), \code{sample_rate}
  (24000), and \code{audio_sec}, like \code{\link{generate}}
}
\description{
Re-synthesizes \code{audio} so the same words and prosody come out in
the target voice (Python chatterbox's \code{ChatterboxVC}). No text
or T3 generation is involved: the source speech is tokenized directly
(25 tokens/s) and S3Gen renders the tokens with the target speaker's
conditioning, so the result follows the source's timing.
}
\examples{
\dontrun{
model <- chatterbox("cuda")
res <- voice_convert(model, "source_speech.wav", "target_voice.wav")
write_audio(res$audio, res$sample_rate, "converted.wav")
}
}
