% tinyrox says don't edit this manually, but it can't stop you!
\name{bytes_to_tensor}
\alias{bytes_to_tensor}
\title{Convert raw bytes to torch tensor}
\usage{
bytes_to_tensor(bytes, dtype, shape, device)
}
\arguments{
\item{bytes}{Raw vector of bytes}

\item{dtype}{Safetensors dtype string}

\item{shape}{Integer vector of tensor dimensions}

\item{device}{Target device}
}
\value{
torch tensor
}
\description{
Convert raw bytes to torch tensor
}
