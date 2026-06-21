% tinyrox says don't edit this manually, but it can't stop you!
\name{resample_audio}
\alias{resample_audio}
\title{Resample audio}
\usage{
resample_audio(samples, from_sr, to_sr)
}
\arguments{
\item{samples}{Numeric vector of audio samples}

\item{from_sr}{Source sample rate}

\item{to_sr}{Target sample rate}
}
\value{
Resampled audio samples
}
\description{
Resample audio
}
\examples{
tone <- sin(2 * pi * 440 * seq(0, 1, length.out = 24000))
tone_16k <- resample_audio(tone, 24000, 16000)
}
