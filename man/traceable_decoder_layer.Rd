% tinyrox says don't edit this manually, but it can't stop you!
\name{traceable_decoder_layer}
\alias{traceable_decoder_layer}
\title{Traceable decoder layer with pre-allocated KV cache}
\usage{
traceable_decoder_layer(layer, max_cache_len = 300L)
}
\arguments{
\item{layer}{Original llama_decoder_layer}

\item{max_cache_len}{Maximum cache length}
}
\value{
nn_module
}
\description{
Traceable decoder layer with pre-allocated KV cache
}
