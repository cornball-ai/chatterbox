% tinyrox says don't edit this manually, but it can't stop you!
\name{create_voice_embedding}
\alias{create_voice_embedding}
\title{Create voice embedding from reference audio}
\usage{
create_voice_embedding(model, audio, sample_rate = NULL, autocast = NULL,
                       norm_loudness = NULL)
}
\arguments{
\item{model}{Chatterbox model}

\item{audio}{Reference audio (file path, numeric vector, or torch tensor)}

\item{sample_rate}{Sample rate of audio (if not a file)}

\item{autocast}{Ignored (kept for API compatibility)}

\item{norm_loudness}{Normalize the reference to -27 LUFS before
conditioning (\code{\link{normalize_loudness}}). Default matches
Python: \code{TRUE} for turbo models, \code{FALSE} for standard.}
}
\value{
Voice embedding that can be used for synthesis
}
\description{
Create voice embedding from reference audio
}
\examples{
\dontrun{
model <- chatterbox("cuda")
voice <- create_voice_embedding(model, "reference_voice.wav")
res <- generate(model, "Reusing a cached voice.", voice)
}
}
