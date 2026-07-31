% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_status}
\alias{resident_status}
\title{Status of a Resident Model}
\usage{
resident_status(res)
}
\arguments{
\item{res}{A `chatterbox_resident` handle.}
}
\value{
A list: variant, state, in_flight, device (the exact bound
  device, e.g. `"cuda:0"`), pinned_bytes, gpu_bytes (logical sums
  over the tensor manifest, not allocator statistics; in the "broken"
  state this is the observed on-GPU manifest bytes, or NA when
  unknowable), identity (per-artifact sha256, HF repo/revision,
  variant), paths (diagnostic), last_error.
}
\description{
Callable in every state, including "broken" and "unloaded".
}
