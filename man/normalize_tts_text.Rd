% tinyrox says don't edit this manually, but it can't stop you!
\name{normalize_tts_text}
\alias{normalize_tts_text}
\title{Normalize text for TTS}
\usage{
normalize_tts_text(text, caps = TRUE, punctuation = TRUE)
}
\arguments{
\item{text}{Character scalar.}

\item{caps}{Apply the internal-caps mitigation. Default TRUE.}

\item{punctuation}{Apply punctuation normalization. Default TRUE.}
}
\value{
Normalized text.
}
\description{
The single normalization entry point. Applies, in order: the
R-specific internal-caps mitigation (\code{normalize_internal_caps}),
then punctuation normalization (\code{punc_norm}: whitespace collapse,
first-letter capitalization, uncommon-punctuation rewrite, trailing
period). \code{punc_norm} is the Python-parity piece; the caps step is
R-only and can be turned off.
}
\examples{
normalize_tts_text("hello   world")
}
