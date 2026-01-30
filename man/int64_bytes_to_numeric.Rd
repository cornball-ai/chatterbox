% tinyrox says don't edit this manually, but it can't stop you!
\name{int64_bytes_to_numeric}
\alias{int64_bytes_to_numeric}
\title{Convert int64 bytes to numeric values}
\usage{
int64_bytes_to_numeric(bytes, n_elements)
}
\arguments{
\item{bytes}{Raw vector of int64 bytes}

\item{n_elements}{Number of int64 values}
}
\value{
Numeric vector
}
\description{
R doesn't have native int64 support, so we read as two int32s and combine.
This works correctly for values up to 2^53 (R's numeric precision limit).
}
