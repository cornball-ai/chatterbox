% tinyrox says don't edit this manually, but it can't stop you!
\name{load_s3gen}
\alias{load_s3gen}
\title{Load S3Gen from safetensors file}
\usage{
load_s3gen(path, device = "cpu")
}
\arguments{
\item{path}{Path to s3gen.safetensors}

\item{device}{Device to load to ("cpu", "cuda", etc.)}
}
\value{
S3Gen model with loaded weights
}
\description{
Load S3Gen from safetensors file
}
