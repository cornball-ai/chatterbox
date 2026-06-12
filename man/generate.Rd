% tinyrox says don't edit this manually, but it can't stop you!
\name{generate}
\alias{generate}
\title{Generate speech from text}
\usage{
generate(model, text, voice, exaggeration = 0.5, cfg_weight = 0.5,
         temperature = 0.8, top_p = 1, min_p = 0.05, autocast = NULL,
         traced = FALSE, backend = c("r", "cpp"), top_k = 1000L,
         repetition_penalty = 1.2, normalize_text = TRUE,
         max_new_tokens = 1000L, max_cache_len = NULL)
}
\arguments{
\item{model}{Chatterbox model}

\item{text}{Text to synthesize}

\item{voice}{Voice embedding from create_voice_embedding() or path to reference audio}

\item{exaggeration}{Emotion/expression exaggeration level (0-1, default 0.5)}

\item{cfg_weight}{Classifier-free guidance weight (higher = more adherence to text, default 0.5)}

\item{temperature}{Sampling temperature (default 0.8)}

\item{top_p}{Top-p (nucleus) sampling threshold. Default 1.0 (disabled),
matching the Python reference.}

\item{min_p}{Minimum probability threshold relative to the most likely
token (default 0.05, matching the Python reference). Standard model only.}

\item{autocast}{Use mixed precision (float16) on CUDA for faster
inference. Default FALSE: the Python reference runs float32, and
float16 output diverges slightly. Opt in for speed on tight VRAM.}

\item{traced}{Logical. Use JIT-traced inference. Default FALSE.}

\item{backend}{Character. Inference backend, either "r" or "cpp".
Default "r". The cpp backend (experimental) runs the decode loop as
compiled C++ on the stable ATen API: with tuned GC settings (see
\code{\link{chatterbox_gc_options}}) it is the fastest native path
and, unlike \code{traced = TRUE}, has no JIT cold start and no
TorchScript dependency.}

\item{top_k}{Integer. Top-k sampling parameter (turbo model only).
Default 1000.}

\item{repetition_penalty}{Numeric. Repetition penalty. Default 1.2.
Applied sign-dependently like HF transformers: positive logits are
divided, negative ones multiplied.}

\item{normalize_text}{Logical. If TRUE (default), pre-process text to
reduce model failure modes: lowercase words with internal capitals
(which the model interprets as emphasis cues and often produces
silent audio for). Set to FALSE to skip case normalization.
Punctuation normalization (whitespace collapse, first-letter
capitalization, trailing period) always runs, matching the Python
reference implementation.}

\item{max_new_tokens}{Maximum speech tokens to generate (default 1000,
= 40 s of audio; the model's own ceiling is 4096).}

\item{max_cache_len}{KV cache positions for the cpp and traced
backends. Default NULL: cpp auto-sizes so generation always fits
(~1 MB VRAM per position); traced keeps its 350-position trace (a
new size triggers a fresh ~50 s JIT compile). Ignored by the pure-R
backend, which has no pre-allocated cache.}
}
\value{
List with elements:
  \describe{
    \item{audio}{Numeric vector of audio samples}
    \item{sample_rate}{Sample rate in Hz}
    \item{eos_found}{Logical. Whether the model emitted an end-of-speech
      token (TRUE) or hit the token cap (FALSE). FALSE often indicates
      garbage output and a need to retry or split the input.}
    \item{n_tokens}{Number of speech tokens generated}
    \item{audio_sec}{Audio duration in seconds}
  }
}
\description{
Generate speech from text
}
