% tinyrox says don't edit this manually, but it can't stop you!
\name{fsmn_multi_head_attention}
\alias{fsmn_multi_head_attention}
\title{FSMN Multi-Head Attention}
\usage{
fsmn_multi_head_attention(n_state, n_head, kernel_size = 31L)
}
\arguments{
\item{n_state}{Hidden dimension}

\item{n_head}{Number of heads}

\item{kernel_size}{FSMN kernel size (default 31)}
}
\value{
nn_module
}
\description{
Multi-head attention with Frequency-domain Self-attention Memory Network
}
