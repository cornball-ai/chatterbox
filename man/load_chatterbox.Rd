% tinyrox says don't edit this manually, but it can't stop you!
\name{load_chatterbox}
\alias{load_chatterbox}
\title{Load Chatterbox model weights}
\usage{
load_chatterbox(model)
}
\arguments{
\item{model}{Chatterbox model object}
}
\value{
Chatterbox model with loaded weights
}
\description{
Load pretrained weights for all model components.
Requires prior download via \code{\link{download_chatterbox_models}}.
Idempotent: an already-loaded model is returned unchanged, so
\code{chatterbox(load = TRUE)} followed by a stray
\code{load_chatterbox()} does not reload.
}
\examples{
\dontrun{
model <- chatterbox("cuda", load = FALSE)
model <- load_chatterbox(model)
}
}
