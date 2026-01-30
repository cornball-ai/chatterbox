% tinyrox says don't edit this manually, but it can't stop you!
\name{precompute_freqs_cis}
\alias{precompute_freqs_cis}
\title{Precompute rotary position embedding frequencies}
\usage{
precompute_freqs_cis(dim, end, theta = 10000)
}
\arguments{
\item{dim}{Dimension (head_dim)}

\item{end}{Maximum sequence length}

\item{theta}{Base frequency}
}
\value{
Complex frequency tensor
}
\description{
Precompute rotary position embedding frequencies
}
