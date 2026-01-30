% tinyrox says don't edit this manually, but it can't stop you!
\name{feed_forward}
\alias{feed_forward}
\title{Feed-forward network for transformer
Matches diffusers FeedForward: net = [GELU(proj), Dropout, Linear]}
\usage{
feed_forward(dim, hidden_dim = NULL)
}
\arguments{
\item{dim}{Input dimension}

\item{hidden_dim}{Hidden dimension (typically 4x dim)}
}
\value{
nn_module
}
\description{
Feed-forward network for transformer
Matches diffusers FeedForward: net = [GELU(proj), Dropout, Linear]
}
