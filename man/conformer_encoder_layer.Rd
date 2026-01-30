% tinyrox says don't edit this manually, but it can't stop you!
\name{conformer_encoder_layer}
\alias{conformer_encoder_layer}
\title{Conformer Encoder Layer}
\usage{
conformer_encoder_layer(
  n_feat = 512,
  n_head = 8,
  n_ffn = 2048,
  dropout_rate = 0.1
)
}
\arguments{
\item{n_feat}{Feature dimension}

\item{n_head}{Number of attention heads}

\item{n_ffn}{Feed-forward hidden dimension}

\item{dropout_rate}{Dropout rate}
}
\value{
nn_module
}
\description{
Single conformer block with attention and feed-forward (no convolution).
}
