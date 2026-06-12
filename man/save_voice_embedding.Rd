% tinyrox says don't edit this manually, but it can't stop you!
\name{save_voice_embedding}
\alias{save_voice_embedding}
\title{Save a voice embedding to disk}
\usage{
save_voice_embedding(voice, path)
}
\arguments{
\item{voice}{Voice embedding from \code{\link{create_voice_embedding}}}

\item{path}{Output file path (suggested extension: .rds-like custom,
e.g. "narrator.voice")}
}
\value{
\code{path}, invisibly
}
\description{
Persists a prepared voice (the R analogue of Python 0.1.7's
\code{Conditionals.save()}) so it can be reused across sessions
without the reference audio or recomputation. Tensors are moved to
CPU before saving; the format is \code{\link[torch]{torch_save}}
(not compatible with Python's \code{.pt} conditionals).
}
