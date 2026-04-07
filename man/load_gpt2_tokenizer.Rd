% tinyrox says don't edit this manually, but it can't stop you!
\name{load_gpt2_tokenizer}
\alias{load_gpt2_tokenizer}
\title{Load GPT-2 BPE tokenizer}
\usage{
load_gpt2_tokenizer(vocab_path, merges_path, added_tokens_path = NULL)
}
\arguments{
\item{vocab_path}{Path to vocab.json}

\item{merges_path}{Path to merges.txt}

\item{added_tokens_path}{Path to added_tokens.json (optional)}
}
\value{
Tokenizer object (list)
}
\description{
Loads vocab.json, merges.txt, and optionally added_tokens.json
for GPT-2 byte-level BPE tokenization.
}
