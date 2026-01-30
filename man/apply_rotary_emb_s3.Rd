% tinyrox says don't edit this manually, but it can't stop you!
\name{apply_rotary_emb_s3}
\alias{apply_rotary_emb_s3}
\title{Apply rotary position embeddings}
\usage{
apply_rotary_emb_s3(xq, xk, freqs_cis)
}
\arguments{
\item{xq}{Query tensor}

\item{xk}{Key tensor}

\item{freqs_cis}{Precomputed frequencies}
}
\value{
List with rotated q and k
}
\description{
Apply rotary position embeddings
}
