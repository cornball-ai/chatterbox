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

# Load model
model <- chatterbox("cuda")
model <- load_chatterbox(model)

# Generate speech
jfk <- system.file("audio", "jfk.mp3", package = "chatterbox")
result <- generate(model, "Hello, this is a test!", jfk)
write_audio(result$audio, result$sample_rate, "output.wav")

# Or one-liner:
quick_tts("Hello world!", "ref.wav", "out.wav")
```
## Differences from the Python implementation

This package targets behavioral parity with chatterbox-tts 0.1.4, with a
few deliberate differences:

- **No audio watermark.** Python chatterbox embeds Resemble's Perth
  imperceptible watermark in every generated clip; this port does not.
  If you need provenance marking for generated audio, add it downstream.
- **A reference voice is required.** Python falls back to a builtin
  default voice (`conds.pt`); the R API asks for reference audio
  explicitly and skips that ~105 MB download.
- **Reliability extras.** `generate()` reports `eos_found`, `n_tokens`,
  and `audio_sec`, normalizes problem text by default
  (`normalize_text = TRUE`), and stops degenerate token loops early.
  Python 0.1.4 (English) generates until the token cap in those cases.
- **Backend token caps.** The pure-R backend generates up to 1000 speech
  tokens (~40 s); `traced = TRUE` and `backend = "cpp"` are limited by
  their pre-allocated KV cache (350 positions including ~80-100 of
  conditioning, so roughly 10 s of audio per call). Long texts:
  use `tts_chunked()`.
- **Voice conversion (`vc.py`) and the multilingual model are not
  ported.**
