% tinyrox says don't edit this manually, but it can't stop you!
\name{chatterbox}
\alias{chatterbox}
\title{Create (and load) a Chatterbox TTS model}
\usage{
chatterbox(device = "cpu", turbo = FALSE, load = TRUE)
}
\arguments{
\item{device}{Device to use ("cpu", "cuda", "mps", etc.)}

\item{turbo}{Use turbo model (GPT-2 backbone, MeanFlow decoder). Default FALSE.}

\item{load}{Load pretrained weights before returning. Default TRUE.
Requires a prior download (\code{\link{download_chatterbox_models}}).}
}
\value{
Chatterbox TTS model object, loaded unless \code{load = FALSE}
}
\description{
Constructs the model object and, by default, loads the pretrained
weights in the same call - the Python reference's
\code{from_pretrained}/\code{from_local} do both at once. Pass
\code{load = FALSE} for the bare object (e.g. to inspect it or test
the not-loaded error paths), then load later with
\code{\link{load_chatterbox}}.
}
