% tinyrox says don't edit this manually, but it can't stop you!
\name{create_voice_embedding}
\alias{create_voice_embedding}
\title{Create voice embedding from reference audio}
\usage{
create_voice_embedding(model, audio, sample_rate = NULL, autocast = NULL)
}
\arguments{
\item{model}{Chatterbox model}

\item{audio}{Reference audio (file path, numeric vector, or torch tensor)}

\item{sample_rate}{Sample rate of audio (if not a file)}

\item{autocast}{Ignored (kept for API compatibility)}
}
\value{
Voice embedding that can be used for synthesis
}
\description{
Create voice embedding from reference audio
}
