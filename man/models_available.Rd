% tinyrox says don't edit this manually, but it can't stop you!
\name{models_available}
\alias{models_available}
\title{Check if Models are Downloaded}
\usage{
models_available(revision = NULL)
}
\arguments{
\item{revision}{Optional exact 40-hex commit to resolve against. With one,
this answers about that snapshot and needs neither a \code{refs/} entry
nor the network.}
}
\value{
TRUE if all model files exist locally
}
\description{
Check if Models are Downloaded
}
\examples{
models_available()
}
