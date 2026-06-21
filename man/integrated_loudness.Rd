% tinyrox says don't edit this manually, but it can't stop you!
\name{integrated_loudness}
\alias{integrated_loudness}
\title{Integrated loudness (ITU-R BS.1770-4)}
\usage{
integrated_loudness(samples, sample_rate)
}
\arguments{
\item{samples}{Numeric vector of mono audio samples.}

\item{sample_rate}{Sample rate in Hz.}
}
\value{
Loudness in LUFS (\code{-Inf} for silence or when no block
  passes the gates).
}
\description{
Measures the integrated gated loudness of a mono signal in LUFS,
matching pyloudnorm's K-weighting meter (the measurement Python
chatterbox turbo applies to reference audio).
}
\examples{
samples <- sin(2 * pi * 440 * seq(0, 1, length.out = 48000))
integrated_loudness(samples, 48000)
}
