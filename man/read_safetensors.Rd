% tinyrox says don't edit this manually, but it can't stop you!
\name{read_safetensors}
\alias{read_safetensors}
\title{Read safetensors file}
\usage{
read_safetensors(path, device = "cpu")
}
\arguments{
\item{path}{Path to .safetensors file}

\item{device}{Device to load tensors to (ignored, CPU only)}
}
\value{
Named list of torch tensors
}
\description{
Loads tensors from a safetensors file as Rtorch tensors.
}
