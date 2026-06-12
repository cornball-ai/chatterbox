% tinyrox says don't edit this manually, but it can't stop you!
\name{generate_batch}
\alias{generate_batch}
\title{Generate speech for several texts with one batched synthesis pass}
\usage{
generate_batch(model, texts, voice, ...)
}
\arguments{
\item{model}{Loaded chatterbox model (standard, not turbo)}

\item{texts}{Character vector of texts to synthesize}

\item{voice}{Shared voice: voice_embedding or reference audio path}

\item{...}{Arguments passed through to the T3 stage, as in
\code{\link{generate}} (exaggeration, cfg_weight, temperature,
top_p, min_p, backend, repetition_penalty, normalize_text,
max_new_tokens, max_cache_len)}
}
\value{
List with one \code{\link{generate}}-style result per text
  (audio, sample_rate, eos_found, n_tokens, audio_sec)
}
\description{
Runs T3 token generation per text (autoregressive, sequential), then
synthesizes ALL utterances in a single batched S3Gen pass (one CFM
solve and one vocoder call over the padded batch). Per-utterance
results match single \code{\link{generate}} calls up to CFM noise
handling - the fixed noise buffer means row i sees the same initial
noise it would alone. Standard model only.
}
