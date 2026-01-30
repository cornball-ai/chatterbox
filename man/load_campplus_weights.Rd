% tinyrox says don't edit this manually, but it can't stop you!
\name{load_campplus_weights}
\alias{load_campplus_weights}
\title{Load CAMPPlus weights from safetensors}
\usage{
load_campplus_weights(model, state_dict, prefix = "speaker_encoder.")
}
\arguments{
\item{model}{CAMPPlus model}

\item{state_dict}{Named list of tensors}

\item{prefix}{Prefix for weight keys (default "speaker_encoder.")}
}
\value{
Model with loaded weights
}
\description{
Load CAMPPlus weights from safetensors
}
