% tinyrox says don't edit this manually, but it can't stop you!
\name{normalize_tts_text}
\alias{normalize_tts_text}
\title{Normalize text for TTS}
\usage{
normalize_tts_text(text)
}
\arguments{
\item{text}{Character scalar.}
}
\value{
Normalized text.
}
\description{
Lowercases words that contain internal capital letters (e.g.
"ALERT", "Rarely"). The Chatterbox model interprets internal capitals
as emphasis cues, which often causes it to produce only the first word
followed by silence. Sentence-initial capitals are left alone.
}
