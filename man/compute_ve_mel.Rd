% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_ve_mel}
\alias{compute_ve_mel}
\title{Compute mel spectrogram for voice encoder}
\usage{
compute_ve_mel(wav, config = voice_encoder_config())
}
\arguments{
\item{wav}{Audio samples (numeric vector)}

\item{config}{Voice encoder config}
}
\value{
Mel spectrogram (batch, time, n_mels)
}
\description{
Uses power spectrum (magnitude^2) without log compression,
matching Python's mel_type="amp" and mel_power=2.0.
}
