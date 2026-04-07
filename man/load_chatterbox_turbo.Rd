% tinyrox says don't edit this manually, but it can't stop you!
\name{load_chatterbox_turbo}
\alias{load_chatterbox_turbo}
\title{Load Chatterbox Turbo model weights}
\usage{
load_chatterbox_turbo(model)
}
\arguments{
\item{model}{Chatterbox model object (with turbo=TRUE)}
}
\value{
Chatterbox model with loaded weights
}
\description{
Loads the turbo variant (GPT-2 backbone, MeanFlow decoder).
Requires prior download via \code{\link{download_chatterbox_turbo_models}}.
}
