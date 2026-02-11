% tinyrox says don't edit this manually, but it can't stop you!
\docType{data}
\name{traceable_attention}
\alias{traceable_attention}
\title{Traceable attention module with pre-allocated KV cache}
\format{
An object of class \code{nn_module_generator} of length 1.
}
\usage{
traceable_attention
}
\description{
This module is designed to be traced with jit_trace. It uses:
- Pre-allocated KV cache of fixed max size
- Attention mask to indicate valid cache positions
- Returns only output tensor (no lists/dicts)
}
\keyword{datasets}
