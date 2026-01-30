% tinyrox says don't edit this manually, but it can't stop you!
\name{apply_rotary_pos_emb}
\alias{apply_rotary_pos_emb}
\title{Apply rotary position embeddings to Q and K}
\usage{
apply_rotary_pos_emb(q, k, cos, sin, position_ids)
}
\arguments{
\item{q}{Query tensor (batch, heads, seq, head_dim)}

\item{k}{Key tensor (batch, heads, seq, head_dim)}

\item{cos}{Cosine cache}

\item{sin}{Sine cache}

\item{position_ids}{Position indices}
}
\value{
List with rotated q and k
}
\description{
Apply rotary position embeddings to Q and K
}
