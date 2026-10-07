% tinyrox says don't edit this manually, but it can't stop you!
\name{campplus}
\alias{campplus}
\title{CAMPPlus speaker encoder}
\usage{
campplus(
  feat_dim = 80,
  embedding_size = 192,
  growth_rate = 32,
  init_channels = 128
)
}
\arguments{
\item{feat_dim}{Input feature dimension (default 80)}

\item{embedding_size}{Output embedding size (default 192)}

\item{growth_rate}{Dense block growth rate (default 32)}

\item{init_channels}{Initial TDNN channels (default 128)}
}
\value{
nn_module
}
\description{
CAMPPlus speaker encoder
}
