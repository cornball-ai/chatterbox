% tinyrox says don't edit this manually, but it can't stop you!
\name{tdnn_layer}
\alias{tdnn_layer}
\title{TDNN Layer}
\usage{
tdnn_layer(
  in_channels,
  out_channels,
  kernel_size,
  stride = 1,
  dilation = 1,
  padding = NULL
)
}
\arguments{
\item{in_channels}{Input channels}

\item{out_channels}{Output channels}

\item{kernel_size}{Kernel size}

\item{stride}{Stride}

\item{dilation}{Dilation}

\item{padding}{Padding (default: computed from kernel_size and dilation)}
}
\value{
nn_module
}
\description{
TDNN Layer
}
