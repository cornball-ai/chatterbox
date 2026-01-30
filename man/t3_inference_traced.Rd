% tinyrox says don't edit this manually, but it can't stop you!
\name{t3_inference_traced}
\alias{t3_inference_traced}
\title{T3 inference with JIT tracing (optimized)}
\usage{
t3_inference_traced(
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

\item{cond}{Conditioning object}

\item{text_tokens}{Text token tensor}

\item{max_new_tokens}{Maximum tokens to generate}

\item{temperature}{Sampling temperature}

\item{cfg_weight}{Classifier-free guidance weight}

\item{top_p}{Top-p sampling threshold}

\item{min_p}{Min-p filtering threshold}

\item{repetition_penalty}{Repetition penalty}

\item{max_cache_len}{Maximum KV cache length}
}
\value{
Generated speech token tensor
}
\description{
Uses jit_trace for ~8x faster per-token inference.
}
