% tinyrox says don't edit this manually, but it can't stop you!
\name{make_pad_mask}
\alias{make_pad_mask}
\title{Create padding mask}
\usage{
make_pad_mask(lengths, max_len = NULL)
}
\arguments{
\item{lengths}{Sequence lengths}

\item{max_len}{Maximum length}
}
\value{
Boolean mask (TRUE for padded positions)
}
\description{
Create padding mask
}
