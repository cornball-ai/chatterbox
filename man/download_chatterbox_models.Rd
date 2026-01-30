% tinyrox says don't edit this manually, but it can't stop you!
\name{download_chatterbox_models}
\alias{download_chatterbox_models}
\title{Download Chatterbox Models from HuggingFace}
\usage{
download_chatterbox_models(force = FALSE)
}
\arguments{
\item{force}{Re-download even if files exist}
}
\value{
Named list of local file paths (invisibly)
}
\description{
Download all Chatterbox model files from HuggingFace.
In interactive sessions, asks for user consent before downloading.
}
\examples{
\dontrun{
# Download models (~2GB)
download_chatterbox_models()
}
}
