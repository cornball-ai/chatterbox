% tinyrox says don't edit this manually, but it can't stop you!
\name{update_kv_cache}
\alias{update_kv_cache}
\title{Update KV cache with new K/V values}
\usage{
update_kv_cache(cache, layer_idx, new_k, new_v, position)
}
\arguments{
\item{cache}{Cache list from create_kv_cache}

\item{layer_idx}{Layer index (1-indexed)}

\item{new_k}{New key tensor (batch, heads, 1, head_dim)}

\item{new_v}{New value tensor (batch, heads, 1, head_dim)}

\item{position}{Current position (0-indexed)}
}
\description{
Update KV cache with new K/V values
}
