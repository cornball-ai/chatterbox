# chatteRbox

Pure R port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) using torch. No Python dependencies - runs entirely in R.

## Architecture

```
chatteRbox
├── Text Tokenizer (BPE)
├── Voice Encoder (speaker embeddings)
├── T3 Model (text → speech tokens)
│   ├── Llama backbone
│   ├── Perceiver resampler
│   └── Attention with KV cache
└── S3Gen (speech tokens → waveform)
    └── HiFi-GAN vocoder
```

## Key Exports

| Function | Purpose |
|----------|---------|
| `chatterbox(device)` | Create model object |
| `load_chatterbox(model)` | Load pretrained weights |
| `tts(model, text, voice)` | Generate speech |
| `create_voice_embedding(model, audio)` | Create speaker embedding |
| `quick_tts(text, ref_audio, output)` | One-liner convenience |

## Usage

```r
library(chatteRbox)

# Load model (downloads weights on first use)
model <- chatterbox("cuda")
model <- load_chatterbox(model)

# Generate speech with voice cloning
result <- tts(model, "Hello world!", "reference_voice.wav")
write_audio(result$audio, result$sample_rate, "output.wav")

# Or one-liner:
quick_tts("Hello!", "ref.wav", "out.wav")
```

## Internal Components

### R/safetensors.R - Pure R Safetensors Reader

Complete implementation without external dependencies:

```r
read_safetensors(path, device = "cpu")
```

Features:
- Parses safetensors header (JSON)
- Handles F16, BF16, F32, I8, I16, I32, I64, U8, BOOL
- Manual IEEE 754 half-precision conversion
- Direct loading to torch tensors
- Handles empty string keys (CRAN safetensors v0.2.0 fails on these)

**Note:** Empty string keys are common in PyTorch forward hook captures. Access by position:
```r
weights <- read_safetensors("model.safetensors")
empty_idx <- which(names(weights) == "")
weights[[empty_idx]]  # Works
weights[[""]]         # Always NULL (R limitation)
```

### R/t3.R - T3 Text-to-Speech Model

Transformer-based text-to-speech with:
- Llama backbone (RoPE, RMSNorm, SwiGLU)
- Perceiver resampler for cross-attention
- KV cache inference for autoregressive generation
- Classifier-free guidance support

### R/llama.R - Llama Model Components

Pure R torch implementation:
- `llama_attention` - Multi-head attention with RoPE
- `llama_mlp` - SwiGLU feed-forward
- `llama_decoder_layer` - Transformer block
- `rms_norm` - RMSNorm layer

### R/hifigan.R - HiFi-GAN Vocoder

Neural vocoder for waveform synthesis:
- Multi-period discriminator
- Multi-scale discriminator
- Residual blocks with dilated convolutions

### R/voice_encoder.R - Speaker Encoder

Voice embedding extraction for cloning.

## Model Weights

Weights are automatically downloaded to `~/.cache/chatterbox/`:
- `t3_cfg.safetensors` - T3 model
- `s3gen.safetensors` - S3Gen vocoder
- `ve.safetensors` - Voice encoder
- `tokenizer.json` - BPE tokenizer

## PyTorch Migration Notes

This package is a complete PyTorch → R torch migration example:

1. **Safetensors loading**: Pure R, no Python
2. **Attention patterns**: Self-attention, cross-attention, perceiver
3. **KV cache inference**: Autoregressive generation
4. **Float16/BF16 handling**: Manual conversion in R

Key lessons:
- Use `torch::with_no_grad()` for inference (not `local_no_grad()`)
- Dimension indexing: PyTorch 0-indexed → R torch 1-indexed
- `$view()` requires contiguous memory, use `$reshape()` when unsure

## Dependencies

- torch (>= 0.12.0)
- tuneR (audio I/O)
- jsonlite (tokenizer parsing)

## Development

```bash
# Using tinyverse toolchain
r -e 'rhydrogen::document(); rhydrogen::install()'
r -e 'tinytest::test_package("chatteRbox")'
```

## Related

- Part of the [cornyverse](https://github.com/cornball-ai/cornyverse) ecosystem
- Alternative to ttsapi for local TTS (no container required)
- pytorch-migration skill for migration patterns
