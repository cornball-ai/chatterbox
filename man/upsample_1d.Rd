% tinyrox says don't edit this manually, but it can't stop you!
\name{upsample_1d}
\alias{upsample_1d}
\title{Upsample 1D}
\usage{
upsample_1d(channels = 512, stride = 2L)
}
\arguments{
\item{channels}{Number of channels}

\item{stride}{Upsample factor}
}
\value{
nn_module
}
\description{
2x upsampling using interpolation + convolution.
}
