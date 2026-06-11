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
Mirrors the HF tokenizers pipeline used by the Python reference:
added tokens ([SPACE], [laughter], [sigh], ...) are extracted first
and map directly to their ids; BPE merges run on the remaining text,
in priority order (first merge = highest priority).
}
