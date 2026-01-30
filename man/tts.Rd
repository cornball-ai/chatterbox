% tinyrox says don't edit this manually, but it can't stop you!
\name{tts}
\alias{tts}
\title{Generate speech from text}
\usage{
tts(
  model,
  text,
  voice,
  exaggeration = 0.5,
  cfg_weight = 0.5,
  temperature = 0.8,
  top_p = 0.9,
  autocast = NULL,
  traced = FALSE
)
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
}
\value{
List with audio (numeric vector) and sample_rate
}
\description{
Generate speech from text
}
