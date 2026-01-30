% tinyrox says don't edit this manually, but it can't stop you!
\name{load_cfm_estimator_weights}
\alias{load_cfm_estimator_weights}
\title{Load CFM estimator weights}
\usage{
load_cfm_estimator_weights(estimator, state_dict, prefix = "")
}
\arguments{
\item{estimator}{CFM estimator module}

\item{state_dict}{State dictionary}

\item{prefix}{Key prefix in state dict}
}
\value{
Number of keys loaded
}
\description{
Load CFM estimator weights
}
\keyword{internal}
