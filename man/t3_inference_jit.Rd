% tinyrox says don't edit this manually, but it can't stop you!
\name{t3_inference_jit}
\alias{t3_inference_jit}
\title{T3 inference with a TorchScript decode loop}
\usage{
t3_inference_jit(model, cond, text_tokens, max_new_tokens = 1000,
                 temperature = 0.8, cfg_weight = 0.5, top_p = 1, min_p = 0.05,
                 repetition_penalty = 1.2, max_cache_len = NULL)
}
\arguments{
\item{model}{T3 model}

\item{cond}{T3 conditioning}

\item{text_tokens}{Tokenized text (tensor)}

\item{max_new_tokens}{Maximum speech tokens to generate}

\item{temperature}{Sampling temperature}

\item{cfg_weight}{Classifier-free guidance weight}

\item{top_p}{Nucleus sampling threshold (1.0 disables, Python default)}

\item{min_p}{Minimum probability threshold}

\item{repetition_penalty}{Repetition penalty}

\item{max_cache_len}{KV cache positions; NULL (default) auto-sizes}
}
\value{
Generated speech tokens (0-indexed), with eos_found attribute
}
\description{
Runs prefill eagerly, then executes each token's 30-layer forward as
a single jit_compile'd TorchScript call. The KV cache is auto-sized
(conditioning + max_new_tokens), so generation always completes;
there is no shape freezing and no per-size recompilation - the
script compiles once per session in milliseconds.
}
