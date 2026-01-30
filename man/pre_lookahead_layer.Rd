% tinyrox says don't edit this manually, but it can't stop you!
\name{pre_lookahead_layer}
\alias{pre_lookahead_layer}
\title{Pre-Lookahead Layer}
\usage{
pre_lookahead_layer(channels = 512, pre_lookahead_len = 3)
}
\arguments{
\item{channels}{Number of channels}

\item{pre_lookahead_len}{Look-ahead length (kernel size - 1 for conv1)}
}
\value{
nn_module
}
\description{
Two causal convolutions with residual connection for look-ahead.
}
