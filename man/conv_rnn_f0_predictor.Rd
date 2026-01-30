% tinyrox says don't edit this manually, but it can't stop you!
\name{conv_rnn_f0_predictor}
\alias{conv_rnn_f0_predictor}
\title{Convolutional RNN F0 Predictor}
\usage{
conv_rnn_f0_predictor(in_channels = 80, cond_channels = 512)
}
\arguments{
\item{in_channels}{Input channels (mel bins)}

\item{cond_channels}{Hidden channels}
}
\value{
nn_module
}
\description{
Convolutional RNN F0 Predictor
}
