% tinyrox says don't edit this manually, but it can't stop you!
\name{upsample_conformer_encoder_full}
\alias{upsample_conformer_encoder_full}
\title{Upsample Conformer Encoder}
\usage{
upsample_conformer_encoder_full(
  input_size = 512,
  output_size = 512,
  num_blocks = 6,
  num_up_blocks = 4,
  n_head = 8,
  n_ffn = 2048,
  dropout_rate = 0.1,
  pre_lookahead_len = 3
)
}
\arguments{
\item{input_size}{Input dimension}

\item{output_size}{Output dimension}

\item{num_blocks}{Number of conformer blocks before upsample}

\item{num_up_blocks}{Number of conformer blocks after upsample}

\item{n_head}{Number of attention heads}

\item{n_ffn}{Feed-forward hidden dimension}

\item{dropout_rate}{Dropout rate}

\item{pre_lookahead_len}{Look-ahead length}
}
\value{
nn_module
}
\description{
Full conformer encoder matching Python UpsampleConformerEncoder.
}
