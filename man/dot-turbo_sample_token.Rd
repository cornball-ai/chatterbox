% tinyrox says don't edit this manually, but it can't stop you!
\name{.turbo_sample_token}
\alias{.turbo_sample_token}
\title{Sample a token using turbo logit processors}
\usage{
.turbo_sample_token(logits, generated_ids, temperature, top_k, top_p,
                    repetition_penalty)
}
\arguments{
\item{logits}{(B, vocab_size) logits}

\item{generated_ids}{Previous token IDs for repetition penalty}

\item{temperature}{Temperature}

\item{top_k}{Top-k}

\item{top_p}{Top-p}

\item{repetition_penalty}{Repetition penalty}
}
\value{
Token tensor (B, 1)
}
\description{
Sample a token using turbo logit processors
}
\keyword{internal}
