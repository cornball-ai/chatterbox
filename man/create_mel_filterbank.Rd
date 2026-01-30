% tinyrox says don't edit this manually, but it can't stop you!
\name{create_mel_filterbank}
\alias{create_mel_filterbank}
\title{Create mel filterbank}
\usage{
create_mel_filterbank(
  sr,
  n_fft,
  n_mels,
  fmin = 0,
  fmax = NULL,
  norm = "slaney",
  htk = FALSE
)
}
\arguments{
\item{sr}{Sample rate}

\item{n_fft}{FFT size}

\item{n_mels}{Number of mel bins}

\item{fmin}{Minimum frequency}

\item{fmax}{Maximum frequency}
}
\value{
Mel filterbank matrix (n_mels x (n_fft/2 + 1))
}
\description{
Create mel filterbank
}
