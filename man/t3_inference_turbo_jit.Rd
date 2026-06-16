% tinyrox says don't edit this manually, but it can't stop you!
\name{t3_inference_turbo_jit}
\alias{t3_inference_turbo_jit}
\title{Turbo T3 inference with a TorchScript decode loop}
\usage{
t3_inference_turbo_jit(model, cond, text_tokens, max_new_tokens = 1000,
                       temperature = 0.8, top_k = 1000L, top_p = 0.95,
                       repetition_penalty = 1.2, max_cache_len = NULL)
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

\item{max_cache_len}{KV cache positions; NULL (default) auto-sizes}
}
\value{
Generated speech tokens (0-indexed), with eos_found attribute
}
\description{
GPT-2 counterpart of \code{\link{t3_inference_jit}}. Runs prefill
eagerly, then executes each token's full GPT-2 forward as a single
jit_compile'd TorchScript call. No CFG (turbo), no rotary embeddings.
The KV cache auto-sizes (conditioning + max_new_tokens), so generation
always completes; the script compiles once per session.
}
