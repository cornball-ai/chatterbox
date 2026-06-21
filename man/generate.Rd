% tinyrox says don't edit this manually, but it can't stop you!
\name{generate}
\alias{generate}
\title{Generate speech from text}
\usage{
generate(model, text, voice, exaggeration = 0.5, cfg_weight = 0.5,
         temperature = 0.8, top_p = 1, min_p = 0.05, autocast = NULL,
         traced = FALSE, backend = c("r", "jit"), top_k = 1000L,
         repetition_penalty = 1.2, normalize_text = FALSE,
         max_new_tokens = 1000L, max_cache_len = NULL, cfm_len = NULL,
         skip_vocoder = FALSE, output_path = NULL)
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

\item{backend}{Character. Inference backend, either "r" or "jit".
Default "r". The jit backend runs each token's full 30-layer
forward as one TorchScript call (compiled once per session, in
milliseconds, via \code{torch::jit_compile}): with tuned GC
settings (see \code{\link{chatterbox_gc_options}}) it is the
fastest native path (~11 ms/token long-form), auto-sizes its KV
cache so generation always completes, and ships no compiled code.
It replaced an equivalent C++ backend (within ~20\\% of its speed)
that required linking against torch's private libraries.}

\item{top_k}{Integer. Top-k sampling parameter (turbo model only).
Default 1000.}

\item{repetition_penalty}{Numeric. Repetition penalty. Default 1.2.
Applied sign-dependently like HF transformers: positive logits are
divided, negative ones multiplied.}

\item{normalize_text}{Logical. Apply the R-specific internal-caps
mitigation (lowercase words with internal capitals). Default FALSE:
it addressed a "first word then silence" failure that was actually
the column-major/STFT bug, now fixed - with the model corrected,
leaving caps intact also preserves intended emphasis (ALL CAPS reads
as emphasis) and acronyms. Set TRUE only if specific text misbehaves.
Punctuation normalization (whitespace collapse, first-letter
capitalization, trailing period) always runs, matching the Python
reference implementation.}

\item{max_new_tokens}{Maximum speech tokens to generate (default 1000,
= 40 s of audio; the model's own ceiling is 4096).}

\item{max_cache_len}{KV cache positions for the jit and traced
backends. Default NULL: jit auto-sizes so generation always fits
(~1 MB VRAM per position); traced keeps its 350-position trace (a
new size triggers a fresh ~50 s trace). Ignored by the pure-R
backend, which has no pre-allocated cache.}

\item{cfm_len}{Optional explicit traced-CFM length (the padded mel
sequence, = 640 + 2 * tokens). Default NULL: the standard traced path
sizes it from the tokens actually generated, rounded up to the
250/500/1000 bucket ladder, so a slow speaker is covered without
guessing from text length. Pass a value to pin it (e.g. to pre-trace
a bucket). Ignored when not traced or for turbo.}

\item{skip_vocoder}{Logical. If TRUE, stop after flow matching and
return the mel spectrogram instead of audio (Python 0.1.7's
\code{skip_vocoder}). The result has a \code{mel} element (tensor,
batch x 80 x frames; 50 frames/s) and no \code{audio}.}

\item{output_path}{Optional WAV path. When set, the audio is also
written there (as a side effect) and the returned list gains a
\code{path} element; the audio is still returned in full. Incompatible
with \code{skip_vocoder} (no audio to write). Default NULL.}
}
\value{
List with elements:
  \describe{
    \item{audio}{Numeric vector of audio samples (omitted when
      \code{skip_vocoder = TRUE}, which returns \code{mel} instead)}
    \item{sample_rate}{Sample rate in Hz}
    \item{eos_found}{Logical. Whether the model emitted an end-of-speech
      token (TRUE) or hit the token cap (FALSE). FALSE often indicates
      garbage output and a need to retry or split the input.}
    \item{n_tokens}{Number of speech tokens generated}
    \item{audio_sec}{Audio duration in seconds}
    \item{path}{Output file path (only when \code{output_path} is set)}
  }
}
\description{
Generate speech from text
}
\examples{
\dontrun{
model <- chatterbox("cuda")
res <- generate(model, "Hello world!", "reference_voice.wav")
write_audio(res$audio, res$sample_rate, "hello.wav")

# Fastest native path: TorchScript decode loop
res <- generate(model, "Hello world!", "reference_voice.wav",
                backend = "jit")
}
}
