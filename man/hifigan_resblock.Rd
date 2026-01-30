% tinyrox says don't edit this manually, but it can't stop you!
\name{hifigan_resblock}
\alias{hifigan_resblock}
\title{HiFiGAN Residual Block}
\usage{
hifigan_resblock(channels = 512, kernel_size = 3, dilations = c(1, 3, 5))
}
\arguments{
\item{channels}{Number of channels}

\item{kernel_size}{Kernel size}

\item{dilations}{List of dilation rates}
}
\value{
nn_module
}
\description{
HiFiGAN Residual Block
}
