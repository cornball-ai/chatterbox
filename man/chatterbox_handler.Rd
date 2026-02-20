% tinyrox says don't edit this manually, but it can't stop you!
\name{chatterbox_handler}
\alias{chatterbox_handler}
\title{Create Chatterbox HTTP Request Handler}
\usage{
chatterbox_handler(device = "cuda", traced = TRUE)
}
\arguments{
\item{device}{Device to use ("cpu" or "cuda"). Default "cuda".}

\item{traced}{Use JIT-traced inference for faster generation. Default TRUE.}
}
\value{
A function with signature
  \code{function(method, path, headers, body)} for use with
  \code{gpu.ctl::serve()}.
}
\description{
Returns a handler function suitable for \code{gpu.ctl::serve()}.
The model is loaded once at handler creation and reused for all requests.
}
\examples{
\dontrun{
handler <- chatterbox_handler()
gpu.ctl::serve(7810, handler)
}
}
