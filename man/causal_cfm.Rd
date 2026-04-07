% tinyrox says don't edit this manually, but it can't stop you!
\name{causal_cfm}
\alias{causal_cfm}
\title{Causal Conditional Flow Matching}
\usage{
causal_cfm(in_channels = 320, out_channels = 80, spk_emb_dim = 80,
           meanflow = FALSE)
}
\arguments{
\item{in_channels}{Input channels (x + mu + spks + cond)}

\item{out_channels}{Output channels (mel bins)}

\item{spk_emb_dim}{Speaker embedding dimension}

\item{meanflow}{Logical. Use mean-flow formulation. Default FALSE.}
}
\value{
nn_module
}
\description{
Causal Conditional Flow Matching
}
