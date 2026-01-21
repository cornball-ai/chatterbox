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
result <- tts(model, "Hello, this is a test!", "reference_audio.wav")
write_audio(result$audio, result$sample_rate, "output.wav")

# Or one-liner:
quick_tts("Hello world!", "ref.wav", "out.wav")
```