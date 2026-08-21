% tinyrox says don't edit this manually, but it can't stop you!
\name{cam_layer}
\alias{cam_layer}
\title{CAM (Context-Aware Masking) Layer}
\usage{
cam_layer(
  bn_channels,
  out_channels,
  kernel_size,
  stride = 1,
  padding = 0,
  dilation = 1,
  reduction = 2
)
}
\arguments{
\item{bn_channels}{Bottleneck channels}

\item{out_channels}{Output channels}

\item{kernel_size}{Kernel size}

\item{stride}{Stride}

\item{padding}{Padding}

\item{dilation}{Dilation}

\item{reduction}{Channel reduction factor}
}
\value{
nn_module
}
\description{
CAM (Context-Aware Masking) Layer
}
