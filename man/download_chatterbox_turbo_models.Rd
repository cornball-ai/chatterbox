% tinyrox says don't edit this manually, but it can't stop you!
\name{download_chatterbox_turbo_models}
\alias{download_chatterbox_turbo_models}
\title{Download Chatterbox Turbo Models from HuggingFace}
\usage{
download_chatterbox_turbo_models(force = FALSE)
}
\arguments{
\item{force}{Re-download even if files exist}
}
\value{
Named list of local file paths (invisibly)
}
\description{
Download all Chatterbox Turbo model files from HuggingFace.
The turbo model uses a GPT-2 backbone and MeanFlow decoder for faster inference.
}
\examples{
\dontrun{
download_chatterbox_turbo_models()
}
}
