% tinyrox says don't edit this manually, but it can't stop you!
\name{chatterbox}
\alias{chatterbox}
\title{Create (and load) a Chatterbox TTS model}
\usage{
chatterbox(device = "cpu", turbo = FALSE, load = TRUE, tune_gc = TRUE)
}
\arguments{
\item{device}{Device to use ("cpu", "cuda", "mps", etc.)}

\item{turbo}{Use turbo model (GPT-2 backbone, MeanFlow decoder). Default FALSE.}

\item{load}{Load pretrained weights before returning. Default TRUE.
Requires a prior download (\code{\link{download_chatterbox_models}}).}

\item{tune_gc}{Tune torch's CUDA GC rates for faster inference (CUDA only,
and only when unset). Persistent session side effect; default TRUE. See
Details.}
}
\value{
Chatterbox TTS model object, loaded unless \code{load = FALSE}
}
\description{
Constructs the model object and, by default, loads the pretrained
weights in the same call - the Python reference's
\code{from_pretrained}/\code{from_local} do both at once. Pass
\code{load = FALSE} for the bare object (e.g. to inspect it or test
the not-loaded error paths), then load later with
\code{\link{load_chatterbox}}.
}
\details{
When \code{tune_gc = TRUE} (the default) and \code{device} is CUDA, this
raises torch's allocator GC floors before the first CUDA op. torch otherwise
runs \code{gc()} on nearly every allocation once a model occupies more than
20\\% of VRAM, which dominates inference. It sets session-global
\code{torch.cuda_allocator_reserved_rate} (the model footprint over VRAM) and
\code{torch.threshold_call_gc}, only when they are unset, so an explicit
setting always wins. This is a deliberate, persistent side effect (torch
reads the rates later, at CUDA init); pass \code{tune_gc = FALSE} to skip it.
}
