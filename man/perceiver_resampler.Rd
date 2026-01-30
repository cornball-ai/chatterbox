% tinyrox says don't edit this manually, but it can't stop you!
\name{perceiver_resampler}
\alias{perceiver_resampler}
\title{Perceiver resampler for conditioning compression}
\usage{
perceiver_resampler(num_query_tokens = 32, embed_dim = 1024, num_heads = 4)
}
\arguments{
\item{num_query_tokens}{Number of query tokens (default 32)}

\item{embed_dim}{Embedding dimension (default 1024)}

\item{num_heads}{Number of attention heads (default 4)}
}
\value{
nn_module
}
\description{
Perceiver resampler for conditioning compression
}
