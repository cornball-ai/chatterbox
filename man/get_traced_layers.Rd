% tinyrox says don't edit this manually, but it can't stop you!
\name{get_traced_layers}
\alias{get_traced_layers}
\title{Get or create traced layers for cached inference}
\usage{
get_traced_layers(model, max_cache_len = 350L)
}
\arguments{
\item{model}{T3 model}

\item{max_cache_len}{Maximum cache length}
}
\value{
List of traced layer modules
}
\description{
Get or create traced layers for cached inference
}
