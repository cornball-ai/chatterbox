# chatterbox

Pure R port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) using torch. No Python dependencies - runs entirely in R.

## Architecture

```
chatterbox
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
library(chatterbox)

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
5. **Conditional Flow Matching (CFM)**: Euler solver for mel generation
6. **HiFi-GAN vocoder**: Conv transpose with various stride/kernel configs

### Critical R torch Differences

**Module callable pattern**: Use `self$submodule$forward(x)` not `self$submodule(x)`:
```r
# WRONG - returns the module object, not output
h <- self$encoder(x)

# CORRECT
h <- self$encoder$forward(x)
```

**torch_arange is inclusive**: R includes both endpoints:
```r
# Python: torch.arange(0, 10) gives [0..9]
# R: torch_arange(0, 10) gives [0..10] - 11 elements!
torch::torch_arange(0, 10 - 1)  # Use this to match Python
```

**torch_clamp may change dtype**: Cast back explicitly:
```r
token <- torch::torch_clamp(token, min = 0L, max = vocab_size - 1L)
token <- token$to(dtype = torch::torch_long())  # Ensure Long for embeddings
```

**R scalar arithmetic promotes dtype**: Use tensor methods:
```r
# WRONG - promotes float16 to float32
y <- x * 0.5

# CORRECT
y <- x$mul(0.5)
```

**conv_transpose1d padding when kernel < stride**:
```r
# Standard: padding = (kernel - stride) / 2
# When kernel=7, stride=8: -0.5 -> Error!

# Fix: use output_padding
if (kernel >= stride) {
  padding <- (kernel - stride) %/% 2
  output_padding <- 0L
} else {
  padding <- 0L
  output_padding <- stride - kernel
}
```

### Flow Matching Implementation

The S3Gen decoder uses Conditional Flow Matching (CFM):
- `solve_euler()` integrates ODE from t=0 to t=1
- CFM estimator predicts velocity field at each timestep
- Classifier-free guidance with batch doubling (cond + uncond)

### HiFi-GAN Vocoder Notes

- Source excitation signal must match mel spectrogram length
- When lengths mismatch, truncate to minimum (may affect quality)
- Multi-period and multi-receptive-field fusion for high quality

## Dependencies

- torch (>= 0.12.0)
- tuneR (audio I/O)
- jsonlite (tokenizer parsing)

## Development

```bash
# Using tinyverse toolchain
r -e 'rhydrogen::document(); rhydrogen::install()'
r -e 'tinytest::test_package("chatterbox")'
```

## Debugging Notes (January 2026)

### Safetensors Partial Download Corruption

**Problem**: t3_cfg.safetensors was 1.3GB instead of 2.1GB due to interrupted download.

**Detection**: Header showed tensor offsets (e.g., `tfmr.norm.weight` at byte 2.1GB) beyond actual file size. Python safetensors error: "incomplete metadata, file not fully covered".

**Fix**: Force re-download with `hf_download(..., force = TRUE)`.

**Prevention**: Check file sizes match HuggingFace metadata, or catch loading errors gracefully.

### Perceiver Architecture Mismatch

**Problem**: R Perceiver had separate cross-attention and self-attention layers. Python reuses a single `attn` module for both operations.

**Python architecture**:
```python
class Perceiver:
    pre_attention_query  # Learnable query tokens
    attn                 # Single AttentionBlock (reused for both cross and self)

class AttentionBlock:
    norm, to_q, to_k, to_v, proj_out  # Standard attention components

# Forward pass:
pre_att = self.attn(query, x)      # Cross-attention (query attends to input)
out = self.attn(pre_att, pre_att)  # Self-attention (query attends to itself)
```

**Wrong R structure** (had separate layers):
```r
# WRONG - separate layers that never got weights loaded
query, norm1, q_proj, k_proj, v_proj, out_proj  # Cross-attention
self_norm, self_q_proj, self_k_proj, ...        # Self-attention (random init!)
```

**Fixed R structure**:
```r
perceiver_resampler <- torch::nn_module(
  initialize = function(num_query_tokens = 32, embed_dim = 1024, num_heads = 4) {
    self$pre_attention_query <- torch::nn_parameter(
      torch::torch_empty(1, num_query_tokens, embed_dim)$uniform_(-v, v)
    )
    self$attn <- attention_block(embed_dim, num_heads)  # Single module!
  },
  forward = function(x) {
    query <- self$pre_attention_query$expand(c(batch_size, -1, -1))
    pre_att <- self$attn$forward(query, x)     # Cross-attention
    out <- self$attn$forward(pre_att, pre_att) # Self-attention (same attn!)
    out
  }
)
```

**Weight key mapping** (Python → R):
```
cond_enc.perceiver.pre_attention_query → $perceiver$pre_attention_query
cond_enc.perceiver.attn.norm.weight    → $perceiver$attn$norm$weight
cond_enc.perceiver.attn.to_q.weight    → $perceiver$attn$to_q$weight
cond_enc.perceiver.attn.to_k.weight    → $perceiver$attn$to_k$weight
cond_enc.perceiver.attn.to_v.weight    → $perceiver$attn$to_v$weight
cond_enc.perceiver.attn.proj_out.weight → $perceiver$attn$proj_out$weight
```

**Result after fix**: Perceiver output std matched (R=0.564 vs Python=0.567).

### Llama Backbone Hidden State Variance (FIXED)

**Status**: RESOLVED. TTS now generates audio with correct EOS detection.

**Root causes found**:

1. **torch_sort returns 1-indexed values in R torch**: When sampling tokens, `torch_sort` returns 1-indexed positions (e.g., 6563 for token 6562). This caused:
   - EOS never detected: comparing 1-indexed (6563) against 0-indexed stop_token (6562)
   - Wrong embeddings: adding +1 to already 1-indexed values

2. **Min-p filtering not recomputing softmax**: After setting `logits[probs < threshold] <- -Inf`, code was sorting original probs instead of recomputing via `nnf_softmax(logits)`.

**Fixes applied**:

```r
# Token sampling fix (R/t3.R)
next_token <- sorted_indices$gather(2L, next_token_idx)

# EOS check - convert 1-indexed to 0-indexed
token_id <- as.integer(next_token$item()) - 1L
if (token_id == config$stop_speech_token) break

# Embedding lookup - sorted_indices already 1-indexed for nn_embedding
next_emb <- model$speech_emb$forward(next_token)  # No $add(1L)

# Return tokens as 0-indexed for downstream
tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)$sub(1L)
```

```r
# Min-p filtering fix (R/t3.R)
probs <- torch::nnf_softmax(logits, dim = -1L)
logits[probs < min_p * max_prob] <- -Inf
probs_filtered <- torch::nnf_softmax(logits, dim = -1L)  # RECOMPUTE!
sorted_result <- torch::torch_sort(probs_filtered, descending = TRUE)
```

**Result**: TTS now generates appropriate audio lengths (4-8 seconds for "Hello world").

**Python comparison using chatterbox container**:
```bash
# Run comparison script inside existing chatterbox container
docker exec -it chatterbox python /tmp/check_hidden.py
```

## Related

- Part of the [cornyverse](https://github.com/cornball-ai/cornyverse) ecosystem
- Alternative to ttsapi for local TTS (no container required)
- pytorch-migration skill for migration patterns
