% tinyrox says don't edit this manually, but it can't stop you!
\name{generate}
\alias{generate}
\title{Generate speech from text}
\usage{
generate(model, text, voice, exaggeration = 0.5, cfg_weight = 0.5,
         temperature = 0.8, top_p = 0.9, autocast = NULL, traced = FALSE,
         backend = c("r", "cpp"), top_k = 1000L, repetition_penalty = 1.2,
         normalize_text = TRUE)
}
\arguments{
\item{model}{Chatterbox model}

\item{text}{Text to synthesize}

\item{voice}{Voice embedding from create_voice_embedding() or path to reference audio}

\item{exaggeration}{Emotion/expression exaggeration level (0-1, default 0.5)}

\item{cfg_weight}{Classifier-free guidance weight (higher = more adherence to text, default 0.5)}

\item{temperature}{Sampling temperature (default 0.8)}

\item{top_p}{Top-p (nucleus) sampling threshold (default 0.9)}

\item{autocast}{Use mixed precision (float16) on CUDA for faster inference (default TRUE on CUDA)}

\item{traced}{Logical. Use JIT-traced inference. Default FALSE.}

\item{backend}{Character. Inference backend, either "r" or "cpp". Default "r".}

\item{top_k}{Integer. Top-k sampling parameter. Default 1000.}

\item{repetition_penalty}{Numeric. Repetition penalty. Default 1.2.}

\item{normalize_text}{Logical. If TRUE (default), pre-process text to
reduce model failure modes: lowercase words with internal capitals
(which the model interprets as emphasis cues and often produces
silent audio for). Set to FALSE to pass text through unchanged.}
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
