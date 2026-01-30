% tinyrox says don't edit this manually, but it can't stop you!
\name{positionwise_feedforward}
\alias{positionwise_feedforward}
\title{Positionwise Feed Forward}
\usage{
positionwise_feedforward(n_feat = 512, n_ffn = 2048, dropout_rate = 0.1)
}
\arguments{
\item{n_feat}{Input/output dimension}

\item{n_ffn}{Hidden dimension}

\item{dropout_rate}{Dropout rate}
}
\value{
nn_module
}
\description{
Two-layer feed-forward network with SiLU activation.
}
