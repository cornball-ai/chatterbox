% tinyrox says don't edit this manually, but it can't stop you!
\name{make_non_pad_mask_s3}
\alias{make_non_pad_mask_s3}
\title{Create non-padding mask}
\usage{
make_non_pad_mask_s3(lengths, max_len)
}
\arguments{
\item{lengths}{Tensor of sequence lengths}

\item{max_len}{Maximum sequence length}
}
\value{
Boolean mask tensor (TRUE for valid positions)
}
\description{
Create non-padding mask
}
