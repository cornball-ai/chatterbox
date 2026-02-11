% tinyrox says don't edit this manually, but it can't stop you!
\name{.get_cpp_weights}
\alias{.get_cpp_weights}
\title{Extract model weights for C++ decode loop}
\usage{
.get_cpp_weights(model)
}
\arguments{
\item{model}{T3 model}
}
\value{
List with layers, final_norm, speech_head, speech_emb, speech_pos_emb
}
\description{
Extracts all layer weights into a flat structure suitable for passing
to the C++ decode function. Just creates list references (no copies).
}
