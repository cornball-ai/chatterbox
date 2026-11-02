% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_generate}
\alias{resident_generate}
\title{Generate Speech with a Resident Model}
\usage{
resident_generate(res, text, voice = NULL, ...)
}
\arguments{
\item{res}{A `chatterbox_resident` handle, currently active.}

\item{text}{Text to synthesize.}

\item{voice}{Reference voice (audio path or voice embedding), per
[generate()]; NULL for the built-in voice of this model's variant
([builtin_voice()]).}

\item{...}{Passed to [generate()]: `exaggeration`,
`cfg_weight`, `temperature`, `backend`, ...}
}
\value{
The generation result, same shape as [generate()].
}
\description{
Runs [generate()] through the resident handle's model.
Requires the model to be active; marks the handle in-flight for the
duration so a concurrent deactivation is refused.
}
