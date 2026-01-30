% tinyrox says don't edit this manually, but it can't stop you!
\name{llama_rms_norm}
\alias{llama_rms_norm}
\title{RMS Normalization module}
\usage{
llama_rms_norm(hidden_size, eps = 1e-05)
}
\arguments{
\item{hidden_size}{Dimension to normalize}

\item{eps}{Epsilon for numerical stability}
}
\value{
nn_module
}
\description{
RMS Normalization module
}
