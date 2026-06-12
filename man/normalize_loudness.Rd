% tinyrox says don't edit this manually, but it can't stop you!
\name{normalize_loudness}
\alias{normalize_loudness}
\title{Normalize audio to a target loudness}
\usage{
normalize_loudness(samples, sample_rate, target_lufs = -27)
}
\arguments{
\item{samples}{Numeric vector of mono audio samples.}

\item{sample_rate}{Sample rate in Hz.}

\item{target_lufs}{Target integrated loudness (default -27, the
Python turbo conditioning default).}
}
\value{
Gain-adjusted samples.
}
\description{
Applies a constant gain so the signal measures \code{target_lufs}
integrated loudness. Mirrors Python chatterbox turbo's
\code{norm_loudness()}: when the gain is non-finite or non-positive
(e.g. silence), the input is returned unchanged.
}
