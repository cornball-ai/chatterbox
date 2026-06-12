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
torch's allocators invoke a full R garbage collection based on
settings that are read ONCE at torch startup. The default CUDA
trigger (collections begin once torch reserves 20 percent of the
card) sits below chatterbox's ~4.6 GB loaded footprint on most GPUs,
which makes autoregressive inference collection-bound: ~91 percent
of pure-R generation wall time is GC, and even compiled-loop backends
are throttled by it (their allocations flow through the same
allocator). With the trigger line above the model floor, pure R runs
~10x faster and the jit backend ~15x.
}
\details{
Only one option matters for speed:
\code{torch.cuda_allocator_reserved_rate}. Sweeps over 0.3-0.8 all
give identical speed; the value just chooses how high the VRAM
plateau sits. \code{torch.cuda_allocator_allocated_rate} (default
0.8) caps that plateau and can be lowered to ~0.6 on shared GPUs at
no speed cost. The remaining settings
(\code{torch.threshold_call_gc},
\code{torch.cuda_allocator_allocated_reserved_rate}) measured as not
worth touching.

This helper does not (and cannot) change the settings for the current
session: torch reads them at initialization, so they belong in your
.Rprofile or at the very top of a script, before torch loads. It
prints the exact snippet for this machine and warns when torch is
already initialized.

Rule of thumb for loops: collect once per utterance, not thousands of
times inside it. \code{\link{tts_chunked}} does this automatically;
in your own batch loops, call \code{gc()} after each
\code{generate()}.

}
\examples{
chatterbox_gc_options(vram_gb = 16)
}
