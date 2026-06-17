# chatterbox

chatterbox is an R package that is an R port of [resemble AI's chatterbox library](https://github.com/resemble-ai/chatterbox). It is written entirely in R using torch and has no Python dependencies.

## Installation

You can install the development version of chatterbox from GitHub with:
```
remotes::install_github("cornball-ai/chatterbox")
```

# Usage

```R
# Set timeout to 10 minutes to allow model download
options(timeout = 600)

library(chatterbox)

# Load model (constructs and loads in one call)
model <- chatterbox("cuda")

# Generate speech
jfk <- system.file("audio", "jfk.mp3", package = "chatterbox")
result <- generate(model, "Hello, this is a test!", jfk)
write_audio(result$audio, result$sample_rate, "output.wav")

# Or one-liner:
quick_tts("Hello world!", "ref.wav", "out.wav")
```

## Serving

`serve()` runs an OpenAI-compatible TTS server (`POST /v1/audio/speech`,
`GET /health`) that loads the model once and stays resident on the GPU:

```r
chatterbox::serve(port = 7810L)               # regular model
chatterbox::serve(port = 7810L, turbo = TRUE) # Turbo (fewer FLOPs; fits a tight VRAM budget)
```

Point any OpenAI-style client at it (e.g. `tts.api::set_tts_base()`). Built on
base R sockets; a systemd unit ships in
`system.file("chatterbox.service", package = "chatterbox")`. That unit is a
template that runs the **regular** model by default — add `turbo = TRUE` to its
`ExecStart` if you want Turbo (e.g. to co-reside with another model on a small
card).

## Differences from the Python implementation

This package targets behavioral parity with chatterbox-tts 0.1.7, with a
few deliberate differences:

- **No audio watermark.** Python chatterbox embeds Resemble's Perth
  imperceptible watermark in every generated clip; this port does not.
  If you need provenance marking for generated audio, add it downstream.
- **A reference voice is required.** Python falls back to a builtin
  default voice (`conds.pt`); the R API asks for reference audio
  explicitly and skips that ~105 MB download.
- **Reliability extras.** `generate()` reports `eos_found`, `n_tokens`,
  and `audio_sec`, always applies Python-parity punctuation
  normalization, and stops degenerate token loops early (Python 0.1.4
  English generates until the token cap in those cases). The R-only
  internal-caps mitigation is opt-in via `normalize_text = TRUE`
  (default `FALSE`; the failure it patched was a since-fixed bug).
- **One-call model load.** `chatterbox("cuda")` constructs *and* loads by
  default; pass `load = FALSE` for the bare object. `load_chatterbox()`
  is idempotent, so older two-step code still works.
- **Backend token caps.** The pure-R and `backend = "jit"` paths
  generate up to `max_new_tokens` (default 1000, ~40 s; jit auto-sizes
  its KV cache so generation always completes). `traced = TRUE` is
  limited by its pre-allocated 350-position cache (roughly 10 s of
  audio per call). Long texts: `tts_chunked()`.
- **GC tuning is automatic and matters a lot.** With torch's default
  allocator settings, autoregressive inference spends most of its wall time
  (~85% on a regular-model run) in R garbage collection. `chatterbox("cuda")`
  tunes torch's CUDA allocator GC by default (`tune_gc = TRUE`), a roughly 6x
  speedup; pass `tune_gc = FALSE` to opt out. `chatterbox_gc_options()` still
  prints the snippet if you prefer to set the `options()` yourself, and the
  performance vignette has the measurements.
- **Voice conversion (`vc.py`) and the multilingual model are not
  ported.**
