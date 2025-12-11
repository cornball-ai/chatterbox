# chatteRbox

An R package that ports the [Chatterbox TTS engine](https://github.com/resemble-ai/chatterbox) to R using torch. Generate high-quality speech with voice cloning and emotion control.

## Features

- **Text-to-Speech**: Convert text to natural-sounding speech at 24kHz
- **Voice Cloning**: Clone any voice from a short reference audio clip
- **Emotion Control**: Adjust expressiveness and emotion intensity
- **Pure R**: No Python dependencies required - runs entirely in R via torch

## Installation

```r
# Install from GitHub
devtools::install_github("troyhernandez/chatteRbox")
```

### Requirements

- R >= 4.0.0
- torch >= 0.12.0 (will be installed automatically)

## Quick Start

```r
library(chatteRbox)

# One-liner TTS (downloads models automatically on first use)
quick_tts("Hello, world!", "output.wav")

# With voice cloning
quick_tts(
  "Hello, this is my cloned voice.",
  "output.wav",
  audio_prompt = "reference_voice.wav"
)
```

## Usage

### Basic Usage

```r
library(chatteRbox)

# Create and load the model
model <- chatterbox()
model <- load_chatterbox(model)

# Generate speech
result <- tts(model, "Hello, this is a test.")

# Save to file
write_audio(result$audio, result$sample_rate, "output.wav")
```

### Voice Cloning

```r
# Create a voice embedding from reference audio
voice <- create_voice_embedding(model, "speaker_reference.wav")

# Generate speech with cloned voice
result <- tts(model, "Now I sound like the reference speaker.", voice = voice)
write_audio(result$audio, result$sample_rate, "cloned_output.wav")
```

### Controlling Expression

```r
result <- tts(
  model,
  text = "This is very exciting news!",
  voice = voice,
  exaggeration = 0.8,   # Higher = more expressive (0-1)
  cfg_weight = 0.5,     # Text adherence strength
  temperature = 0.8     # Randomness in generation
)
```

### Long Text

For longer texts, use chunked generation which splits by sentences:

```r
result <- tts_chunked(
  model,
  "This is a longer piece of text. It will be split into sentences.
   Each sentence is synthesized separately and concatenated.",
  voice = voice
)
```

### Saving Directly to File

```r
tts_to_file(
  model,
  "Save this speech directly to a file.",
  filename = "speech.wav",
  voice = voice
)
```

## Model Downloads

Models are downloaded automatically on first use. They are cached in `~/.cache/chatterbox` by default.

To manually download models:

```r
download_chatterbox_models()
```

To use a different cache directory:

```r
Sys.setenv(CHATTERBOX_CACHE = "/path/to/cache")
```

## GPU Support

The package supports CPU, CUDA, and MPS (Apple Silicon) devices:

```r
# Use GPU if available
model <- chatterbox(device = "cuda")
model <- load_chatterbox(model)

# Or Apple Silicon
model <- chatterbox(device = "mps")
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `chatterbox()` | Create a Chatterbox model instance |
| `load_chatterbox()` | Load model weights |
| `quick_tts()` | One-liner TTS with automatic model loading |
| `tts()` | Generate speech from text |
| `tts_to_file()` | Generate speech and save to WAV |
| `tts_chunked()` | Generate long texts in chunks |

### Voice Functions

| Function | Description |
|----------|-------------|
| `create_voice_embedding()` | Create speaker embedding from audio |
| `compute_speaker_embedding()` | Extract speaker embedding |

### Audio Utilities

| Function | Description |
|----------|-------------|
| `read_audio()` | Load audio from WAV file |
| `write_audio()` | Save audio to WAV file |
| `resample_audio()` | Resample audio to different rate |

### Model Management

| Function | Description |
|----------|-------------|
| `download_chatterbox_models()` | Download all model files |
| `models_available()` | Check if models are downloaded |

## Architecture

chatteRbox uses a multi-stage neural pipeline:

```
Text → Tokenizer → T3 Model → Speech Tokens → S3Gen → Mel Spectrogram → HiFiGAN → Audio
                      ↑
              Voice Encoder
              (speaker embedding)
```

- **T3 Model**: Llama-based transformer (520M params) that converts text tokens to speech tokens
- **S3Gen**: Conformer + Conditional Flow Matching decoder for mel spectrogram generation
- **HiFiGAN**: Neural vocoder with source-filter architecture for waveform synthesis
- **Voice Encoder**: LSTM-based model for extracting speaker embeddings

## Credits

This package is an R port of [Chatterbox](https://github.com/resemble-ai/chatterbox) by [Resemble AI](https://www.resemble.ai/).

## License

MIT
