% tinyrox says don't edit this manually, but it can't stop you!
\name{tokenize_text}
\alias{tokenize_text}
\title{Encode text to token IDs using BPE}
\usage{
tokenize_text(tokenizer, text)
}
\arguments{
\item{tokenizer}{Tokenizer object}

\item{text}{Input text}
}
\value{
Integer vector of token IDs
}
\description{
Implements proper BPE (Byte Pair Encoding) using the merge list.
Merges are applied in priority order (first merge = highest priority).
}
