# chatterbox

Pure R port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) using torch. No Python dependencies - runs entirely in R.

## Python Reference

**Target version**: chatterbox-tts 0.1.4 (PyPI)
**Reference container**: `chatterbox-tts:blackwell` (cornball-ai/chatterbox-tts-api)
- Built: Dec 5, 2025
- CUDA: 12.8.1 (Blackwell-compatible)
- Python: 3.11 with venv at `/app/.venv`

Use this container for validation/comparison:
```bash
docker run --rm --gpus all \
  -v ~/chatterbox/scripts:/scripts \
  -v ~/chatterbox/outputs:/outputs \
  chatterbox-tts:blackwell \
  python /scripts/your_script.py
```

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

# Use traced inference for 2.2x faster generation
result <- tts(model, "Hello world!", "reference_voice.wav", traced = TRUE)

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

**Tensor methods return new tensors, not in-place**: Unlike Python's `tensor.sub_()`, R torch methods return a new tensor:
```r
# WRONG - tokens is unchanged!
tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)
tokens$sub(1L)  # Returns new tensor, but result is discarded
tokens  # Still has original values!

# CORRECT - assign the result
tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)
tokens <- tokens$sub(1L)  # Now tokens has the subtracted values
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
# CRITICAL: Must assign result - $sub() returns new tensor, doesn't modify in place!
tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)
tokens <- tokens$sub(1L)  # Convert 1-indexed to 0-indexed
```

```r
# Min-p filtering fix (R/t3.R)
probs <- torch::nnf_softmax(logits, dim = -1L)
logits[probs < min_p * max_prob] <- -Inf
probs_filtered <- torch::nnf_softmax(logits, dim = -1L)  # RECOMPUTE!
sorted_result <- torch::torch_sort(probs_filtered, descending = TRUE)
```

**Result**: TTS now generates appropriate audio lengths (4-8 seconds for "Hello world").

### T3 Token Return Bug (Silent Output)

**Problem**: TTS produced 19 seconds of silence despite all components being individually validated.

**Root cause**: In `t3_inference()`, the token conversion to 0-indexed was not assigned:
```r
# BUG - $sub() returns new tensor, original unchanged!
tokens <- torch::torch_cat(predicted, dim = 2)$squeeze(1)
tokens$sub(1L)  # Result discarded, tokens still 1-indexed

# FIX - assign the result
tokens <- tokens$sub(1L)
```

**Effect**: All speech tokens passed to S3Gen were off-by-one (1-6563 instead of 0-6562), causing:
- Every token mapped to wrong embedding (token N used embedding N+1)
- Valid max token 6560 appeared as 6561, filtered as invalid
- S3Gen received garbage embeddings, producing silence

**Verification**: S3Gen with correct 0-indexed tokens produces audio with std=0.048 (vs 0.0005 when broken).

**Python comparison using chatterbox container**:
```bash
# Run comparison script inside existing chatterbox container
docker exec -it chatterbox python /tmp/check_hidden.py
```

## Validation Status (January 2026)

Systematic component-by-component validation against Python reference.

### Summary

| Component | Status | Max Diff | Test Script |
|-----------|--------|----------|-------------|
| Mel Spectrogram | ✅ Validated | < 0.000001 | `scripts/test_voice_encoder.R` |
| Voice Encoder | ✅ Validated | < 0.000256 | `scripts/test_voice_encoder.R` |
| S3 Tokenizer | ✅ Validated | 100% match | `scripts/test_s3tokenizer.R` |
| T3 Conditioning | ✅ Validated | < 0.000002 | `scripts/test_t3_cond.R` |
| T3 Llama Backbone | ✅ Validated | < 0.00003 | `scripts/test_t3_llama.R` |
| T3 Token Generation | ✅ Fixed | N/A | See debugging notes |
| CAMPPlus Speaker Encoder | ✅ Validated | < 0.001471 | `scripts/test_campplus.R` |
| Conformer Encoder | ✅ Validated | < 0.0004 | `scripts/test_encoder_steps.R` |
| CFM Estimator | ✅ Validated | < 0.052 | `scripts/test_estimator_euler_inputs.R` |
| CFM Decoder | ✅ Validated | < 0.028 | `scripts/test_cfm_full.R` |
| HiFi-GAN Vocoder | ✅ Validated | < 0.026 | `scripts/test_hifigan.R` |

**Current state**: All components fully validated. End-to-end TTS pipeline complete.

### ✅ Mel Spectrogram (Voice Encoder)

**Validated**: R matches Python with max diff < 0.000001

**Parameters** (voice encoder mel):
- n_mels: 40
- sample_rate: 16000 Hz
- n_fft: 400
- hop_length: 160
- fmin: 0, fmax: 8000
- mel_type: "amp" (power spectrum, no log)
- mel_power: 2.0

**Fixes applied**:
1. **Mel filterbank formula**: Changed from HTK to Slaney (librosa default)
   - Slaney: linear below 1000 Hz, log above
   - HTK: `2595 * log10(1 + hz/700)` everywhere
2. **STFT padding**: Changed from `(n_fft - hop_size) / 2 = 120` to `n_fft // 2 = 200`
   - librosa `center=True` pads `n_fft // 2` on each side
3. **Filterbank normalization**: Use Hz bandwidth, not mel bandwidth

### ✅ Voice Encoder (Speaker Embedding)

**Validated**: R matches Python with max diff < 0.000256

**Architecture**:
- 3-layer LSTM (input: 40 mels, hidden: 256)
- Linear projection (256 → 256)
- ReLU activation
- L2 normalization

**Process**:
1. Compute mel spectrogram (40 mels, 16kHz)
2. Split into overlapping 160-frame partials (50% overlap, frame_step=80)
3. LSTM forward on all partials
4. Take final hidden state from layer 3
5. Project → ReLU → L2 normalize each partial
6. Average all partial embeddings
7. L2 normalize final embedding

**Test script**: `scripts/test_voice_encoder.R`

### ✅ T3 Conditioning (Embeddings & Perceiver)

**Validated**: R matches Python with max diff < 0.000002

**Components validated**:
- Text token embedding: 0.000000 diff
- Text position embedding: 0.000000 diff
- Speech start embedding: 0.000000 diff
- Speaker projection: 0.000000 diff
- Perceiver resampler: 0.000002 diff (32 query tokens)
- Emotion projection: 0.000000 diff
- Full conditioning: 0.000002 diff
- Full input embeddings: 0.000002 diff

**Architecture**:
- Conditioning: 34 positions (1 speaker + 32 perceiver + 1 emotion)
- Text vocab: 704 tokens
- Speech vocab: 8194 tokens
- Embedding dim: 1024
- start_speech_token: 6561
- stop_speech_token: 6562
- Llama backbone: 520M parameters

**Test script**: `scripts/test_t3_cond.R`

### ✅ T3 Llama Backbone (Transformer Forward)

**Validated**: R matches Python with max diff < 0.00003

**Components validated**:
- Hidden states: [1, 43, 1024] - max diff 0.000029
- Speech logits: [1, 1, 8194] - max diff 0.000027

**Statistics comparison**:
| Metric | Python | R |
|--------|--------|---|
| Hidden mean | 0.023360 | 0.023360 |
| Hidden std | 1.170491 | 1.170491 |
| Logits mean | -3.105268 | -3.105261 |
| Logits std | 4.000566 | 4.000564 |

**Test script**: `scripts/test_t3_llama.R`

### ✅ T3 Token Generation (Sampling + EOS)

**Status**: Fixed and working

The T3 autoregressive token generation loop was debugged and fixed. Key issues resolved:

1. **R torch 1-indexing for torch_sort**: `sorted_indices` returns 1-indexed values
2. **Min-p filtering**: Must recompute softmax after setting logits to -Inf
3. **EOS detection**: Convert 1-indexed token back to 0-indexed for comparison

**Result**: TTS generates appropriate audio lengths (4-8 seconds for "Hello world")

See "Llama Backbone Hidden State Variance (FIXED)" in Debugging Notes for details.

### ✅ CAMPPlus Speaker Encoder

**Validated**: R matches Python with max diff < 0.001471

**Architecture**:
- FCM head: Conv2d layers reducing frequency 80→10
- xvector: TDNN blocks with stats pooling
- Output: 192-dim speaker embedding

**Input**: Raw audio at 16kHz
**Output**: L2-normalized 192-dim embedding

**Test script**: `scripts/test_campplus.R`

### ✅ Conformer Encoder (UpsampleConformerEncoder)

**Validated**: R matches Python with max diff < 0.0004

**Architecture**:
- LinearNoSubsampling input embedding
- EspnetRelPositionalEncoding (position 0 at center of buffer)
- 6 Conformer encoder layers
- 2x upsample layer (interpolate + conv1d)
- 4 Conformer up-encoder layers
- Final LayerNorm

**Fixes applied**:
1. **Positional encoding center**: Changed from `max_len - 1` to `max_len` (R 1-indexed)
2. **PE buffer creation**: Matched Python's flip + concat structure for relative positions
3. **rel_shift function**: Complete rewrite to match Python's reshape + slice approach

**Test script**: `scripts/test_encoder_steps.R`

### ✅ CFM Decoder (CausalConditionalCFM)

**Validated**: R matches Python with max diff < 0.028

**Architecture**:
- CFM Estimator: UNet-style ConditionalDecoder (71.3M parameters)
  - SinusoidalPosEmb + TimestepEmbedding for time conditioning
  - 1 down block: CausalResnetBlock1D + 4 BasicTransformerBlocks + CausalConv1d
  - 12 mid blocks: each with CausalResnetBlock1D + 4 BasicTransformerBlocks
  - 1 up block: CausalResnetBlock1D + 4 BasicTransformerBlocks + CausalConv1d
  - Final: CausalBlock1D + Conv1d projection
- 10-step Euler ODE solver (cosine time schedule)
- Classifier-free guidance (cfg_rate=0.7)

**Fixes applied**:
1. **Up block order**: Changed from `conv -> concat -> resnet -> transformers` to `concat -> resnet -> transformers -> conv` (matching Python)
2. **Attention inner_dim**: Fixed from 256 to 512 (8 heads × 64 head_dim)
3. **GELU with projection**: FeedForward uses internal projection before GELU (diffusers style)
4. **res_conv always present**: Python always has conv layer even when channels match

**Test scripts**: `scripts/test_estimator_euler_inputs.R`, `scripts/test_cfm_full.R`

### ✅ S3Gen (Speech Tokens → Audio)

**Status**: All components validated (encoder, CFM decoder, HiFiGAN vocoder).

**Weights**: `~/.cache/huggingface/hub/models--ResembleAI--chatterbox/` → `s3gen.safetensors` (1.06 GB)

**Architecture** (from Python tracing):
```
S3Token2Wav (inherits S3Token2Mel)
├── tokenizer: S3Tokenizer (✅ validated 100% match)
├── speaker_encoder: CAMPPlus (192-dim speaker embedding)
│   ├── head: FCM (Conv2d layers, freq reduction 80→10)
│   └── xvector: Sequential (TDNN blocks, StatsPool, Dense)
├── mel_extractor: function (24kHz mel spectrogram)
├── flow: CausalMaskedDiffWithXvec
│   ├── input_embedding: Embedding(6561, 512)
│   ├── spk_embed_affine_layer: Linear(192→80)
│   ├── encoder: UpsampleConformerEncoder (6 layers + 2x upsample)
│   ├── encoder_proj: Linear(512→80)
│   └── decoder: CausalConditionalCFM (U-Net style, 10 Euler steps)
│       └── estimator: ConditionalDecoder (down/mid/up blocks)
└── mel2wav: HiFTGenerator
    ├── f0_predictor: ConvRNNF0Predictor
    ├── m_source: SourceModuleHnNSF (harmonic + noise)
    └── decode: Conv + ResBlocks → waveform
```

**Pipeline**:
1. **embed_ref**(ref_wav, ref_sr) → ref_dict:
   - Resample to 24kHz and 16kHz
   - `prompt_feat`: 24kHz mel spectrogram [B, T, 80]
   - `embedding`: CAMPPlus speaker embedding [B, 192]
   - `prompt_token`: S3Tokenizer tokens [B, T//2]

2. **flow.inference**(token, token_len, **ref_dict) → mel:
   - Normalize & project speaker embedding: [B, 192] → [B, 80]
   - Concat prompt_token + speech_tokens → embed
   - UpsampleConformerEncoder: [B, T, 512] → [B, 2T, 512]
   - Project to mel dim: [B, 2T, 512] → [B, 2T, 80]
   - CFM decoder (10 Euler steps): → [B, 80, 2T]

3. **mel2wav.inference**(mel) → waveform:
   - F0 prediction from mel
   - Source signal generation (harmonic + noise)
   - HiFi-GAN decode: [B, 80, T] → [B, T*320]

**Key parameters**:
- Output sample rate: 24kHz
- Token-to-mel ratio: 2x (upsampled in encoder)
- pre_lookahead_len: 3 tokens (trimmed when finalize=False)
- CFM timesteps: 10 (cosine schedule)
- Speaker embedding: 192-dim (CAMPPlus)
- Mel channels: 80

**CAMPPlus Speaker Encoder**:
```
Input: raw audio (16kHz, 1D tensor)
↓
extract_feature: torchaudio Kaldi.fbank(num_mel_bins=80)
  → [B, T, 80] (mean-normalized)
↓
permute: [B, T, 80] → [B, 80, T]
↓
FCM head:
  unsqueeze: [B, 1, 80, T]
  conv1+bn1+relu: [B, 32, 80, T]
  layer1: [B, 32, 40, T]  (freq/2)
  layer2: [B, 32, 20, T]  (freq/2)
  conv2+bn2+relu: [B, 32, 10, T]  (freq/2)
  reshape: [B, 320, T]
↓
xvector:
  tdnn: [B, 320, T] → [B, 128, T/2]
  block1+transit1: [B, 256, T/2]
  block2+transit2: [B, 512, T/2]
  block3+transit3: [B, 512, T/2]
  out_nonlinear: [B, 512, T/2]
  stats (mean+std pool): [B, 1024]
  dense: [B, 192]
```

**Test scripts**:
- `scripts/save_s3gen_detailed.py` - Architecture exploration
- `scripts/save_s3gen_components.py` - Component intermediates
- `scripts/save_campplus_extract.py` - CAMPPlus step-by-step
- `scripts/save_cfm_details.py` - CFM flow tracing

### ✅ S3 Tokenizer

**Validated**: R matches Python with 100% token match (150/150)

**Parameters** (CORRECTED from PseudoCode.md):
- n_mels: 128 (not 80)
- sample_rate: 16000 Hz (not 24kHz)
- n_fft: 400
- hop_length: 160
- output: 25 tokens/second
- codebook_size: 6561 (3^8 FSQ)

**Architecture**:
- AudioEncoderV2 (Whisper-style, 6 layers, 20 heads, 1280 dim)
- FSQVectorQuantization (Finite Scalar Quantization)
- FSMN attention blocks (depthwise conv for temporal context)

**Test script**: `scripts/test_s3tokenizer.R`

### ✅ HiFi-GAN Vocoder

**Validated**: R matches Python with max diff < 0.026

**Architecture** (HiFTGenerator):
- F0 predictor: 5-layer ConvNet (80 → 512 → 1)
- Source module: Sine generator with 8 harmonics + noise
- Upsampling: 3 ConvTranspose1d layers (strides 8, 5, 3)
- Source fusion: Downsample source STFT + ResBlocks at each stage
- ISTFT synthesis: n_fft=16, hop_len=4
- Total upsample factor: 8 * 5 * 3 * 4 = 480x

**Weight normalization**: Python uses ParametrizedConv1d with weight norm.
State dict keys use `parametrizations.weight.original0` (magnitude) and `original1` (direction).
Effective weight: `w = g * v / ||v||`

**Components validated**:
- F0 predictor: max diff 0.000430
- conv_pre: max diff 0.000995
- Full decode: max diff 0.025823

**Test script**: `scripts/test_hifigan.R`

## Performance

See `vignettes/performance.md` for detailed comparison.

### Comparison: Native R vs Container

| Implementation | Precision | Time (~5s audio) | Real-time Factor |
|----------------|-----------|------------------|------------------|
| Container (Python) | float16 | ~2.2s | **2.7x** |
| Native R (traced) | float32 | ~15.6s | **0.35x** |
| Native R (normal) | float32 | ~34s | **0.15x** |

**Traced inference (`traced = TRUE`) is 2.2x faster** than normal R inference.

The container is still ~8x faster due to:
1. **float16 vs float32** - Half the memory bandwidth and compute
2. **Python C++ bindings** - Lower per-operation overhead
3. **Fused kernels** - Python has optimized attention/matmul fusions

### JIT Trace Optimization (Jan 2026)

The `traced = TRUE` parameter compiles transformer layers and KV projectors to C++ graphs
using `torch::jit_trace()`, eliminating R-to-C++ boundary overhead.

**How it works:**
1. Pre-allocate KV cache to max length (350 tokens) with attention mask
2. Trace each decoder layer and KV projector once (first call)
3. Generation loop: update cache in R, run traced forward pass

**Speedup:** 2.2x faster (0.35x real-time vs 0.15x normal)

**Limitations:**
- Cache limited to 350 tokens (including conditioning ~50-100 tokens)
- First call has compilation overhead (~5s to trace 30 layers)
- Long texts may be truncated to fit cache

### Optimizations Applied (Jan 2026)

Improved from 232ms/token to **135ms/token** (42% faster):

1. **SDPA (Scaled Dot-Product Attention)**: Fused attention kernel, 2.7x faster in isolation
   ```r
   # Access unexported function from torch namespace
   sdpa <- get("torch_scaled_dot_product_attention", envir = asNamespace("torch"))
   ```

2. **Vectorized repetition penalty**: O(1) scatter vs O(n) loop
   ```r
   unique_ids <- unique(as.integer(generated_ids$cpu()))
   logits[1, unique_ids] <- logits[1, unique_ids] / penalty
   ```

3. **Single softmax for min-p**: Filter in probability space, renormalize

4. **CPU-first weight loading**: Load to CPU, copy to model, gc(), then move to CUDA
   - Peak VRAM: 6GB → 3.3GB (matches theoretical 3.2GB for 798M params)

### Main Bottleneck

T3 autoregressive token generation (~135ms per token). Each iteration:
- Transformer forward pass through 30 layers with KV cache
- Logit processing (softmax, top-p sampling)
- R-to-C++ boundary crossing for every tensor operation

### When to Use Native R

- No Docker available or desired
- Long-running R sessions (model stays cached, avoids 17s load time)
- Custom fine-tuning or LoRA experimentation
- Full control over inference parameters
- Offline operation required

### When to Use Container

- Speed is critical (10x faster)
- Short-lived scripts (avoid model load overhead)
- Production deployments
- GPU resource management via gpu.ctl

## Related

- Part of the [cornyverse](https://github.com/cornball-ai/cornyverse) ecosystem
- Alternative to tts.api container backend for local TTS (no Docker required)
- Use `tts.api::speech(..., backend = "native")` for unified interface
- pytorch-migration skill for migration patterns
