% tinyrox says don't edit this manually, but it can't stop you!
\name{create_kv_cache}
\alias{create_kv_cache}
\title{Create pre-allocated KV cache}
\usage{
create_kv_cache(batch_size, n_layers, n_heads, head_dim, max_len, device)
}
\arguments{
\item{batch_size}{Batch size}

\item{n_layers}{Number of transformer layers}

\item{n_heads}{Number of attention heads}

\item{head_dim}{Head dimension}

\item{max_len}{Maximum sequence length}

\item{device}{Device to allocate on}
}
\value{
List with k_cache, v_cache, valid_mask
}
\description{
Create pre-allocated KV cache
}
