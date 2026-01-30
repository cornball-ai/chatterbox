% tinyrox says don't edit this manually, but it can't stop you!
\name{load_s3tokenizer_weights}
\alias{load_s3tokenizer_weights}
\title{Load S3Tokenizer weights from state dictionary}
\usage{
load_s3tokenizer_weights(model, state_dict, prefix = "tokenizer.")
}
\arguments{
\item{model}{S3Tokenizer model}

\item{state_dict}{Named list of tensors}

\item{prefix}{Prefix for weight keys (default "tokenizer.")}
}
\value{
Model with loaded weights
}
\description{
Load S3Tokenizer weights from state dictionary
}
