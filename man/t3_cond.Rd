% tinyrox says don't edit this manually, but it can't stop you!
\name{t3_cond}
\alias{t3_cond}
\title{Create T3 conditioning object}
\usage{
t3_cond(
  speaker_emb,
  cond_prompt_speech_tokens = NULL,
  cond_prompt_speech_emb = NULL,
  emotion_adv = 0.5
)
}
\arguments{
\item{speaker_emb}{Speaker embedding tensor (B, 256)}

\item{cond_prompt_speech_tokens}{Optional speech tokens for conditioning}

\item{cond_prompt_speech_emb}{Optional pre-computed speech embeddings}

\item{emotion_adv}{Emotion/exaggeration control (0-1)}
}
\value{
List representing T3Cond
}
\description{
Create T3 conditioning object
}
