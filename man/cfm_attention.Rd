% tinyrox says don't edit this manually, but it can't stop you!
\name{cfm_attention}
\alias{cfm_attention}
\title{Self-attention for transformer block}
\usage{
cfm_attention(dim, num_heads = 8L, head_dim = 64L)
}
\arguments{
\item{dim}{Hidden dimension}

\item{num_heads}{Number of attention heads}

\item{head_dim}{Head dimension (default 64)}
}
\value{
nn_module
}
\description{
Self-attention for transformer block
}
