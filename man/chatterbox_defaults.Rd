% tinyrox says don't edit this manually, but it can't stop you!
\name{chatterbox_defaults}
\alias{chatterbox_defaults}
\title{Recommended chatterbox settings for this machine}
\usage{
chatterbox_defaults(vram_gb = NULL)
}
\arguments{
\item{vram_gb}{Total GPU memory in GB. Default: detected via
nvidia-smi; 0 (or detection failure) means CPU-only. Cards under
5 GB are treated as CPU: the loaded model alone needs ~4.6 GB.}
}
\value{
An object of class \code{"chatterbox_defaults"}: a list with
  \code{device}, \code{vram_gb}, \code{options} (for
  \code{do.call(options, ...)} before torch loads), \code{backend},
  \code{max_new_tokens}, \code{chunk_chars}, and \code{measured}.
}
\description{
Detects the GPU (or its absence) and returns everything worth setting
for it: the torch GC options (which must be set BEFORE torch loads -
see \code{\link{chatterbox_gc_options}} for why), the fastest
validated backend, the per-call token budget, and when to switch to
\code{\link{tts_chunked}}. Printing the result shows a ready-to-paste
setup snippet.
}
\details{
Measured tiers (long-form, tuned GC): 16 GB RTX 5060 Ti - jit
11 ms/token, container parity; 6 GB GTX 1660 Ti - jit 35-38 ms/token
vs container 30, in 4.7 GB VRAM. The 8 and 12 GB tiers are projected
from the rule (the GC trigger line must clear the ~4.6 GB loaded
model) and marked as such when printed.

}
\examples{
chatterbox_defaults(vram_gb = 6)
chatterbox_defaults(vram_gb = 0) # CPU
}
