% tinyrox says don't edit this manually, but it can't stop you!
\name{read_audio}
\alias{read_audio}
\title{Read audio file}
\usage{
read_audio(path)
}
\arguments{
\item{path}{Path to audio file (WAV or MP3 format)}
}
\value{
List with samples (numeric vector normalized to \[-1, 1\]) and sr (sample rate)
}
\description{
Read audio file
}
