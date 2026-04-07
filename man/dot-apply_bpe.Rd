% tinyrox says don't edit this manually, but it can't stop you!
\name{.apply_bpe}
\alias{.apply_bpe}
\title{Apply BPE merges to a token}
\usage{
.apply_bpe(token, bpe_ranks)
}
\arguments{
\item{token}{Character string (already byte-encoded)}

\item{bpe_ranks}{Merge rank map}
}
\value{
Character vector of BPE pieces
}
\description{
Apply BPE merges to a token
}
\keyword{internal}
