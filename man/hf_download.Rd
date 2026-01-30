% tinyrox says don't edit this manually, but it can't stop you!
\name{hf_download}
\alias{hf_download}
\title{Download file from HuggingFace Hub}
\usage{
hf_download(repo_id, filename, force = FALSE)
}
\arguments{
\item{repo_id}{Repository ID (e.g., "ResembleAI/chatterbox")}

\item{filename}{Filename to download}

\item{force}{Re-download even if file exists}
}
\value{
Local path to downloaded file
}
\description{
Download file from HuggingFace Hub
}
