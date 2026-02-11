% tinyrox says don't edit this manually, but it can't stop you!
\docType{data}
\name{cfm_estimator}
\alias{cfm_estimator}
\title{CFM Estimator (ConditionalDecoder)}
\format{
An object of class \code{nn_module_generator} of length 1.
}
\usage{
cfm_estimator
}
\description{
UNet-style architecture with:
- 1 down block (320 -> 256, with 4 transformer blocks)
- 12 mid blocks (256 -> 256, each with 4 transformer blocks)
- 1 up block (512 -> 256 with skip connection, 4 transformer blocks)
}
\keyword{datasets}
