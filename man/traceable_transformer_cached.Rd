% tinyrox says don't edit this manually, but it can't stop you!
\name{traceable_transformer_cached}
\alias{traceable_transformer_cached}
\title{Traceable transformer for cached inference}
\usage{
traceable_transformer_cached(tfmr, max_cache_len = 300L)
}
\arguments{
\item{tfmr}{Original llama_model}

\item{max_cache_len}{Maximum cache length}
}
\value{
nn_module
}
\description{
This wraps the full Llama model for traced cached inference.
Uses pre-allocated KV cache for all layers.
}
