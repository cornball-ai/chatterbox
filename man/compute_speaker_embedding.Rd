% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_speaker_embedding}
\alias{compute_speaker_embedding}
\title{Compute speaker embedding from audio}
\usage{
compute_speaker_embedding(model, audio, sr, overlap = 0.5)
}
\arguments{
\item{model}{Voice encoder model}

\item{audio}{Audio samples (numeric vector or tensor)}

\item{sr}{Sample rate of audio}

\item{overlap}{Overlap between partials (default 0.5)}
}
\value{
Speaker embedding tensor (1, 256)
}
\description{
Compute speaker embedding from audio
}
