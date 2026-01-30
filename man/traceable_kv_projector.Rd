% tinyrox says don't edit this manually, but it can't stop you!
\name{traceable_kv_projector}
\alias{traceable_kv_projector}
\title{Traceable K/V projection module}
\usage{
traceable_kv_projector(layer)
}
\arguments{
\item{layer}{Original llama_decoder_layer}
}
\value{
nn_module
}
\description{
Computes K and V projections with RoPE for a single layer.
Returns concatenated K and V for easy unpacking.
}
