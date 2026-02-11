% tinyrox says don't edit this manually, but it can't stop you!
\docType{data}
\name{traceable_transformer_cached}
\alias{traceable_transformer_cached}
\title{Traceable transformer for cached inference}
\format{
An object of class \code{nn_module_generator} of length 1.
}
\usage{
traceable_transformer_cached
}
\description{
This wraps the full Llama model for traced cached inference.
Uses pre-allocated KV cache for all layers.
}
\keyword{datasets}
