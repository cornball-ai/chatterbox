% tinyrox says don't edit this manually, but it can't stop you!
\name{builtin_voice}
\alias{builtin_voice}
\title{The built-in voice}
\usage{
builtin_voice(turbo = FALSE, device = "cpu")
}
\arguments{
\item{turbo}{Logical. The turbo model's voice (\code{TRUE}) or the
standard model's (\code{FALSE}, the default). A model object may be
given instead, and its variant and device are used.}

\item{device}{Device to hold the tensors on. Ignored when a model is
given.}
}
\value{
A \code{voice_embedding}, as \code{\link{create_voice_embedding}}
  returns. Its \code{sample_rate} is \code{NA}: there is no reference
  clip behind it. Loaded once per model set and device.
}
\description{
Python chatterbox falls back to a default voice when no reference audio
is given: conditionals computed once from a reference clip and shipped
with the weights as \code{conds.pt}. This package ships the same
conditionals, rewritten as safetensors under \code{inst/voices}: one
set for the standard model and one for turbo, which differ (turbo's
conditioning prompt is 375 speech tokens to the standard model's 150).
\code{\link{generate}} and the functions built on it use this voice
when \code{voice} is \code{NULL}.
}
\examples{
\dontrun{
model <- chatterbox("cuda")
res <- generate(model, "No reference clip was given.")
dim(builtin_voice(model)$cond_prompt_speech_tokens)
}
}
