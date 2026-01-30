% tinyrox says don't edit this manually, but it can't stop you!
\name{rel_position_attention}
\alias{rel_position_attention}
\title{Relative Position Multi-Headed Attention}
\usage{
rel_position_attention(n_head = 8, n_feat = 512, dropout_rate = 0.1)
}
\arguments{
\item{n_head}{Number of attention heads}

\item{n_feat}{Feature dimension}

\item{dropout_rate}{Dropout rate}
}
\value{
nn_module
}
\description{
Multi-head attention with relative positional encodings.
}
