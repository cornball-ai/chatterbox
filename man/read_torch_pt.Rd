% tinyrox says don't edit this manually, but it can't stop you!
\name{read_torch_pt}
\alias{read_torch_pt}
\title{Load PyTorch .pt file (for conds.pt)}
\usage{
read_torch_pt(path, device = "cpu")
}
\arguments{
\item{path}{Path to .pt file}

\item{device}{Device to load to}
}
\value{
Loaded object (typically a list/dict structure)
}
\description{
This is a simplified loader for the specific structure of conds.pt
Full pickle parsing is complex; this handles the common case
}
