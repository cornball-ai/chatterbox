% tinyrox says don't edit this manually, but it can't stop you!
\name{tts_to_file}
\alias{tts_to_file}
\title{Generate speech and save to file}
\usage{
tts_to_file(model, text, voice, output_path, ...)
}
\arguments{
\item{model}{Chatterbox model}

\item{text}{Text to synthesize}

\item{voice}{Voice embedding or path to reference audio}

\item{output_path}{Output file path (WAV format)}

\item{...}{Additional arguments passed to generate()}
}
\value{
Invisibly returns a list with elements: \code{path},
  \code{eos_found}, \code{n_tokens}, \code{audio_sec}. When iterating
  over many texts, collect these into a data.frame to identify which
  inputs failed (\code{eos_found = FALSE}) and need reprocessing.
}
\description{
Generate speech and save to file
}
