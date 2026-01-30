% tinyrox says don't edit this manually, but it can't stop you!
\name{read_safetensors}
\alias{read_safetensors}
\title{Read safetensors file}
\usage{
read_safetensors(path, device = "cpu")
}
\arguments{
\item{path}{Path to .safetensors file}

\item{device}{Device to load tensors to ("cpu", "cuda", etc.)}
}
\value{
Named list of torch tensors
}
\description{
Read safetensors file
}
