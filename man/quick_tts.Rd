% tinyrox says don't edit this manually, but it can't stop you!
\name{quick_tts}
\alias{quick_tts}
\title{Quick TTS - one-line text-to-speech}
\usage{
quick_tts(text, reference_audio, output_path = NULL, device = "cpu",
          autocast = NULL, turbo = FALSE)
}
\arguments{
\item{text}{Text to synthesize}

\item{reference_audio}{Path to reference audio file}

\item{output_path}{Optional output file path. If NULL, returns audio data.}

\item{device}{Device to use}

\item{autocast}{Use mixed precision (float16) on CUDA (default TRUE on CUDA)}

\item{turbo}{Logical. Use turbo architecture. Default FALSE.}
}
\value{
The \code{\link{generate}} result list (audio, sample_rate,
  ...). When \code{output_path} is set the audio is also written there
  (the list gains a \code{path} element) and the list is returned
  invisibly so the audio vector does not print.
}
\description{
Loads model if needed and generates speech. Convenient for quick tests.
}
