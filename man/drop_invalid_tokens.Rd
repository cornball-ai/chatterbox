% tinyrox says don't edit this manually, but it can't stop you!
\name{drop_invalid_tokens}
\alias{drop_invalid_tokens}
\title{Drop invalid speech tokens}
\usage{
drop_invalid_tokens(tokens)
}
\arguments{
\item{tokens}{Token tensor or integer vector (0-indexed values)}
}
\value{
Filtered tokens, same type as input
}
\description{
Python parity: slice to the span after the first SOS and before the
first EOS (s3tokenizer drop_invalid_tokens), then drop any remaining
out-of-vocab ids (tts.py filters < SPEECH_VOCAB_SIZE on top). The
old elementwise filter kept post-EOS garbage when generation hit the
token cap with a mid-stream EOS.
}
