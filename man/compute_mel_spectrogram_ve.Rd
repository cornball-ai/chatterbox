% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_mel_spectrogram_ve}
\alias{compute_mel_spectrogram_ve}
\title{Compute mel spectrogram for voice encoder (40 bins, 16kHz)}
\usage{
compute_mel_spectrogram_ve(y, sr = 16000)
}
\arguments{
\item{y}{Audio samples}

\item{sr}{Sample rate (should be 16000)}
}
\value{
Mel spectrogram (batch, time, 40)
}
\description{
Compute mel spectrogram for voice encoder (40 bins, 16kHz)
}
