% tinyrox says don't edit this manually, but it can't stop you!
\name{learned_position_embeddings}
\alias{learned_position_embeddings}
\title{Learned position embeddings module}
\usage{
learned_position_embeddings(seq_len, model_dim, init_std = 0.02)
}
\arguments{
\item{seq_len}{Maximum sequence length}

\item{model_dim}{Embedding dimension}

\item{init_std}{Initialization standard deviation}
}
\value{
nn_module
}
\description{
Learned position embeddings module
}
