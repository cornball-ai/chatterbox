% tinyrox says don't edit this manually, but it can't stop you!
\name{t3_inference_cpp}
\alias{t3_inference_cpp}
\title{T3 inference with C++ decode loop}
\usage{
t3_inference_cpp(
  model,
  cond,
  text_tokens,
  max_new_tokens = 1000,
  temperature = 0.8,
  cfg_weight = 0.5,
  top_p = 0.95,
  min_p = 0.05,
  repetition_penalty = 1.2,
  max_cache_len = 350L
)
}
\arguments{
\item{model}{T3 model}

\item{cond}{T3 conditioning}

\item{text_tokens}{Tokenized text (tensor)}

\item{max_new_tokens}{Maximum speech tokens to generate}

\item{temperature}{Sampling temperature}

\item{cfg_weight}{Classifier-free guidance weight}

\item{top_p}{Nucleus sampling threshold}

\item{min_p}{Minimum probability threshold}

\item{repetition_penalty}{Repetition penalty}

\item{max_cache_len}{Maximum KV cache length}
}
\value{
Generated speech tokens (0-indexed integer vector)
}
\description{
Runs prefill in R, then delegates the entire autoregressive decode loop
to C++ via a single .Call(). Eliminates R->lantern->libtorch per-op
dispatch overhead (~270 matmuls + ~100 elementwise ops per token).
}
