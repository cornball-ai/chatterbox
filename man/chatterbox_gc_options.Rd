% tinyrox says don't edit this manually, but it can't stop you!
\name{chatterbox_gc_options}
\alias{chatterbox_gc_options}
\title{Recommended torch garbage-collection settings for chatterbox}
\usage{
chatterbox_gc_options(vram_gb = NULL)
}
\arguments{
\item{vram_gb}{Total GPU memory in GB. Default: detected via
nvidia-smi, falling back to 16.}
}
\value{
Invisibly, a named list of the recommended options.
}
\description{
torch's allocators invoke a full R garbage collection based on four
settings that are read ONCE at torch startup. The defaults are tuned
for small GPUs and make autoregressive inference collection-bound:
profiling shows ~91 percent of pure-R generation wall time spent in
GC, and tuned settings run the same generation ~12x faster.
}
\details{
This helper does not (and cannot) change the settings for the current
session: torch reads them at initialization, so they belong in your
.Rprofile or at the very top of a script, before torch loads. It
prints the exact snippet for this machine and warns when torch is
already initialized.

The two untouched companion settings
(\code{torch.cuda_allocator_allocated_rate} and
\code{torch.cuda_allocator_allocated_reserved_rate}, both 0.8) are
the self-regulation backstop that forces collections near the top of
the card; leave them at their defaults.

Rule of thumb for loops: collect once per utterance, not thousands of
times inside it. \code{\link{tts_chunked}} does this automatically;
in your own batch loops, call \code{gc()} after each
\code{generate()}.

}
\examples{
chatterbox_gc_options(vram_gb = 16)
}
