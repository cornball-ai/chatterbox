% tinyrox says don't edit this manually, but it can't stop you!
\name{compute_xvector_embedding}
\alias{compute_xvector_embedding}
\title{Compute xvector speaker embedding from audio}
\usage{
compute_xvector_embedding(model, audio, sr)
}
\arguments{
\item{model}{CAMPPlus model}

\item{audio}{Audio samples (tensor or numeric)}

\item{sr}{Sample rate}
}
\value{
Speaker embedding (1, 192)
}
\description{
Compute xvector speaker embedding from audio
}
