% tinyrox says don't edit this manually, but it can't stop you!
\name{read_audio}
\alias{read_audio}
\title{Read audio file}
\usage{
read_audio(path)
}
\arguments{
\item{path}{Path to audio file (WAV or MP3 format)}
}
\value{
List with samples (numeric vector normalized to \[-1, 1\]) and sr (sample rate)
}
\description{
Read audio file
}
\examples{
tmp <- file.path(tempdir(), "tone.wav")
write_audio(sin(2 * pi * 440 * seq(0, 1, length.out = 24000)), 24000, tmp)
a <- read_audio(tmp)
str(a)  # list(samples = ..., sr = ...)
}
