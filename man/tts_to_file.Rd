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
Invisibly returns the output path
}
\description{
Generate speech and save to file
}
