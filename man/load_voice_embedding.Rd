% tinyrox says don't edit this manually, but it can't stop you!
\name{load_voice_embedding}
\alias{load_voice_embedding}
\title{Load a voice embedding from disk}
\usage{
load_voice_embedding(path, device = "cpu")
}
\arguments{
\item{path}{File written by \code{\link{save_voice_embedding}}}

\item{device}{Device to load tensors to (default "cpu"; use the
model's device, e.g. "cuda", for generation)}
}
\value{
A voice_embedding object
}
\description{
Load a voice embedding from disk
}
\examples{
\dontrun{
voice <- load_voice_embedding("narrator.voice", device = "cuda")
model <- chatterbox("cuda")
res <- generate(model, "Loaded a saved voice.", voice)
}
}
