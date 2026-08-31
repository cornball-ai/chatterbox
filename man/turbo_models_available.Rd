% tinyrox says don't edit this manually, but it can't stop you!
\name{turbo_models_available}
\alias{turbo_models_available}
\title{Check if Turbo Models are Downloaded}
\usage{
turbo_models_available(revision = NULL)
}
\arguments{
\item{revision}{Optional exact 40-hex commit to resolve against. With one,
this answers about that snapshot and needs neither a \code{refs/} entry
nor the network.}
}
\value{
TRUE if all turbo model files exist locally
}
\description{
Check if Turbo Models are Downloaded
}
\examples{
turbo_models_available()
}
