% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_load}
\alias{resident_load}
\title{Load Chatterbox as a Resident (Pinned Host Weights)}
\usage{
resident_load(turbo = FALSE, device = "cuda", tune_gc = TRUE, verbose = TRUE)
}
\arguments{
\item{turbo}{Load the turbo variant (default FALSE).}

\item{device}{Target CUDA device for activation (default "cuda").
Residency requires CUDA; pinned host memory exists to feed it. A bare
"cuda" is resolved to the current device's explicit index at load
time, and the handle stays bound to that exact device (`"cuda:N"`)
for every later transition and synchronize.}

\item{tune_gc}{Apply the allocator GC options before CUDA init
(default TRUE), as [chatterbox()] does.}

\item{verbose}{Print progress messages.}
}
\value{
A `chatterbox_resident` handle (an environment). Fields of
  interest via [resident_status()]: state, byte counts, and a
  content identity (per-artifact sha256, HF repo and snapshot revision,
  variant).
}
\description{
Loads the model (standard or turbo) and retains its canonical weights
as page-locked (pinned) CPU tensors, so the GPU representation can be
created and destroyed repeatedly without reloading from disk.
[resident_activate()] copies the weights to the GPU (a DMA
transfer from pinned memory); [resident_deactivate()] frees the
GPU copy and repoints the model at the pinned host storage. The handle
starts inactive.
}
\details{
Weights stay at their native dtypes (the validated numerics are fp32);
nothing is converted. The torch allocator GC options are applied before
the first CUDA operation, exactly as [chatterbox()] does --
without them, both inference and the activation copy itself become
collection-bound on cards where the model exceeds 20 percent of VRAM.

}
\examples{
\dontrun{
res <- resident_load(turbo = TRUE)
resident_activate(res)
out <- resident_generate(res, "Hello from a resident model.",
    system.file("audio", "jfk.wav", package = "chatterbox"))
resident_deactivate(res) # VRAM freed, weights stay pinned in RAM
resident_activate(res) # fast: DMA copy, no disk
resident_unload(res)
}
}
