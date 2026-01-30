% tinyrox says don't edit this manually, but it can't stop you!
\name{init_cache_from_first}
\alias{init_cache_from_first}
\title{Initialize cache with first token K/V values}
\usage{
init_cache_from_first(cache, past_key_values)
}
\arguments{
\item{cache}{Cache list from create_kv_cache}

\item{past_key_values}{List of K/V from first forward pass}
}
\value{
Updated cache (and seq_len as attribute)
}
\description{
Initialize cache with first token K/V values
}
