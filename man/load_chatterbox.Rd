% tinyrox says don't edit this manually, but it can't stop you!
\name{load_chatterbox}
\alias{load_chatterbox}
\title{Load Chatterbox model weights}
\usage{
load_chatterbox(model, compiled = FALSE)
}
\arguments{
\item{model}{Chatterbox model object}

\item{compiled}{If TRUE, compile eligible sub-modules using
\code{Rtorch::compile()} for fused kernel execution.}
}
\value{
Chatterbox model with loaded weights
}
\description{
Load pretrained weights for all model components.
Requires prior download via \code{\link{download_chatterbox_models}}.
}
