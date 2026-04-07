% tinyrox says don't edit this manually, but it can't stop you!
\name{t3_inference_turbo}
\alias{t3_inference_turbo}
\title{Run T3 turbo inference to generate speech tokens}
\usage{
t3_inference_turbo(model, cond, text_tokens, max_new_tokens = 1000,
                   temperature = 0.8, top_k = 1000L, top_p = 0.95,
                   repetition_penalty = 1.2)
}
\arguments{
\item{model}{T3 turbo model}

\item{cond}{T3 conditioning}

\item{text_tokens}{Tokenized text (tensor)}

\item{max_new_tokens}{Maximum speech tokens to generate}

\item{temperature}{Sampling temperature}

\item{top_k}{Top-k sampling}

\item{top_p}{Nucleus sampling threshold}

\item{repetition_penalty}{Repetition penalty}
}
\value{
Generated speech tokens (0-indexed)
}
\description{
Uses GPT-2 backbone without CFG. Simpler sampling with top-k and
repetition penalty.
}
