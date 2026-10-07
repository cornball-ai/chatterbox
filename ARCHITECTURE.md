# Chatterbox Architecture

Reference for the port: how the pipeline fits together, what each component
is, the record of validating each one against the Python reference, and the
bugs found on the way. `CLAUDE.md` holds the working facts an agent needs
on every session; this file is for reading before changing a model file.

## Pipeline Overview

```
Text ──► [1. Text Tokenizer] ──► text_tokens
                                      │
Reference Audio ──► [2. Voice Encoder] ──► speaker_emb (256-dim)
        │                                       │
        │                                       ▼
        └──► [3. S3 Tokenizer] ──► prompt_tokens ──► [4. T3 Model] ──► speech_tokens
                                                           │
Reference Audio ──► [5. CAMPPlus] ──► xvector (192-dim)    │
        │                                   │              │
        └──► [6. Mel Extractor] ──► prompt_mel            │
                                        │                  │
                                        ▼                  ▼
                              [7. Conformer Encoder] ◄── speech_tokens
                                        │
                                        ▼
                              [8. CFM Decoder] ──► mel_spectrogram
                                        │
                                        ▼
                              [9. HiFi-GAN Vocoder] ──► waveform (24kHz)
```

## Components

| # | Component | Purpose | Input | Output | Params |
|---|-----------|---------|-------|--------|--------|
| 1 | Text Tokenizer | BPE text encoding | text string | token IDs (0-703) | - |
| 2 | Voice Encoder | Speaker embedding for T3 | 16kHz audio | 256-dim embedding | ~2M |
| 3 | S3 Tokenizer | Speech tokenization | 16kHz audio | token IDs (0-6560) | ~300M |
| 4 | T3 Model | Text → speech tokens | text + conditioning | speech tokens | 520M |
| 5 | CAMPPlus | Speaker embedding for S3Gen | 16kHz audio | 192-dim xvector | ~7M |
| 6 | Mel Extractor | Reference mel spectrogram | 24kHz audio | [T, 80] mel | - |
| 7 | Conformer Encoder | Speech token encoding | tokens + xvector | [2T, 512] features | ~25M |
| 8 | CFM Decoder | Flow matching → mel | encoder output | [80, 2T] mel | ~71M |
| 9 | HiFi-GAN Vocoder | Mel → waveform | mel spectrogram | 24kHz audio | ~14M |

## Validation record (January 2026)

Each component was compared with the Python reference on the same inputs.
The comparison scripts (`scripts/test_*.R` and the Python capture scripts)
were removed in PR #18 once the port was complete; the figures stand as the
record. Sorted by max difference, largest first:

| # | Component | Max diff |
|---|-----------|----------|
| 8 | CFM Decoder | 0.028 (estimator alone 0.052) |
| 9 | HiFi-GAN Vocoder | 0.026 |
| 5 | CAMPPlus | 0.0015 |
| 7 | Conformer Encoder | 0.0004 |
| 2 | Voice Encoder | 0.00026 |
| 4 | T3 Llama backbone | 0.00003 |
| 4 | T3 Conditioning | 0.000002 |
| 6 | Mel Extractor | 0.000001 |
| 3 | S3 Tokenizer | 0 (150/150 tokens) |
| 1 | Text Tokenizer | 0 (same tokenizer.json) |

Numerically, chatterbox-tts 0.1.7 matches 0.1.4 on the fp32 English path
(mu/spks/cond byte-identical, June 2026); the only change is embed_ref dtype
casting, a no-op in fp32.

### Mel spectrogram (voice encoder)

n_mels 40, 16 kHz, n_fft 400, hop 160, fmin 0, fmax 8000, power spectrum
(no log), mel_power 2.0. Three fixes brought it to < 1e-6: Slaney filterbank
(linear below 1 kHz, log above; not HTK), STFT padding `n_fft // 2 = 200`
(librosa `center=True`), filterbank normalized by Hz bandwidth.

### Voice encoder

3-layer LSTM (40 → 256), linear 256 → 256, ReLU, L2 norm. Mel is split into
160-frame partials at 50% overlap (frame_step 80); the final hidden state of
layer 3 is projected, normalized per partial, averaged, and normalized again.

### T3 conditioning and Llama backbone

34 conditioning positions (1 speaker + 32 perceiver + 1 emotion). Text vocab
704, speech vocab 8194, embedding dim 1024, start_speech_token 6561,
stop_speech_token 6562, backbone 520M parameters. Hidden states
[1, 43, 1024] matched to 2.9e-5, speech logits [1, 1, 8194] to 2.7e-5.

The Perceiver reuses one `attn` block for cross-attention (query attends to
input) and then self-attention (output attends to itself). The first R
version had separate cross and self layers; the self layers never received
weights. Weight keys: `cond_enc.perceiver.pre_attention_query` and
`cond_enc.perceiver.attn.{norm,to_q,to_k,to_v,proj_out}.weight`. After the
fix the perceiver output std matched (R 0.564, Python 0.567).

### CAMPPlus speaker encoder

Kaldi fbank (80 bins, mean-normalized) → FCM head (Conv2d, frequency 80 → 10,
reshape to [B, 320, T]) → xvector (TDNN 320 → 128 at T/2, three
block+transit stages to 512, mean+std stats pool to 1024, dense to 192),
L2-normalized.

### S3 tokenizer

n_mels 128 (not 80), 16 kHz (not 24), n_fft 400, hop 160, 25 tokens/s,
codebook 6561 (3^8 FSQ). Whisper-style AudioEncoderV2 (6 layers, 20 heads,
1280 dim) with FSMN attention blocks and finite scalar quantization.

### Conformer encoder (UpsampleConformerEncoder)

LinearNoSubsampling embedding, EspnetRelPositionalEncoding with position 0
at the center of the buffer, 6 Conformer layers, 2x upsample (interpolate +
conv1d), 4 up-encoder layers, final LayerNorm. Fixes: PE center at `max_len`
(1-indexed), PE buffer built with Python's flip + concat, `rel_shift`
rewritten to match the reshape + slice.

### CFM decoder (CausalConditionalCFM)

Estimator is a UNet-style ConditionalDecoder (71.3M): SinusoidalPosEmb +
TimestepEmbedding; 1 down block (CausalResnetBlock1D + 4
BasicTransformerBlocks + CausalConv1d), 12 mid blocks, 1 up block, final
CausalBlock1D + Conv1d. 10 Euler steps on a cosine schedule, classifier-free
guidance at cfg_rate 0.7 by batch doubling. Fixes: up-block order
`concat → resnet → transformers → conv`, attention inner_dim 512 (8 × 64),
FeedForward projection before GELU (diffusers style), `res_conv` always
present.

### S3Gen pipeline

```
S3Token2Wav
├── tokenizer: S3Tokenizer
├── speaker_encoder: CAMPPlus
├── mel_extractor: 24 kHz mel
├── flow: CausalMaskedDiffWithXvec
│   ├── input_embedding: Embedding(6561, 512)
│   ├── spk_embed_affine_layer: Linear(192 → 80)
│   ├── encoder: UpsampleConformerEncoder
│   ├── encoder_proj: Linear(512 → 80)
│   └── decoder: CausalConditionalCFM
└── mel2wav: HiFTGenerator (f0_predictor, m_source, decode)
```

1. `embed_ref(ref_wav, ref_sr)`: resample to 24 kHz and 16 kHz; `prompt_feat`
   [B, T, 80] mel, `embedding` [B, 192], `prompt_token` [B, T/2].
2. `flow.inference`: normalize and project the speaker embedding to 80,
   concat prompt and speech tokens, encoder [B, T, 512] → [B, 2T, 512],
   project to mel, CFM decode → [B, 80, 2T].
3. `mel2wav.inference`: F0 from mel, harmonic + noise source, HiFi-GAN
   decode [B, 80, T] → [B, T × 320].

Output 24 kHz; token-to-mel ratio 2x; `pre_lookahead_len` 3 tokens (trimmed
when `finalize = FALSE`); source excitation is truncated to the mel length
when they disagree.

### HiFi-GAN vocoder (HiFTGenerator)

F0 predictor (5-layer ConvNet 80 → 512 → 1), sine source with 8 harmonics +
noise, 3 ConvTranspose1d stages (strides 8, 5, 3) with source STFT fusion
and ResBlocks, ISTFT synthesis (n_fft 16, hop 4), total upsample 480x.
Python stores weight-normalized convs as `parametrizations.weight.original0`
(magnitude g) and `original1` (direction v); effective `w = g * v / ||v||`.
F0 predictor matched to 4.3e-4, conv_pre to 1e-3, full decode to 0.026.

## Resolved port bugs

Kept so they are not reintroduced.

- **Partial safetensors download.** `t3_cfg.safetensors` was 1.3 GB instead
  of 2.1 GB after an interrupted download. The header's tensor offsets ran
  past the file size (Python: "incomplete metadata, file not fully covered").
  `download_chatterbox_models(force = TRUE)` re-downloads.
- **Perceiver with separate attention layers.** See T3 conditioning above.
- **1-indexed `torch_sort` in sampling.** `sorted_indices` are 1-indexed, so
  EOS was never detected (6563 compared with 6562) and embeddings were off by
  one. The sampler gathers from `sorted_indices`, feeds the 1-indexed token
  to `nn_embedding` directly, and subtracts 1 for the EOS check and the
  returned tokens.
- **Min-p filtering without recomputing softmax.** After
  `logits[probs < min_p * max_prob] <- -Inf`, sort `nnf_softmax(logits)`,
  not the original probabilities.
- **Dropped assignment on `$sub()`.** `tokens$sub(1L)` without assignment
  left every speech token 1-indexed; S3Gen then read the wrong embeddings and
  produced 19 s of silence (output std 0.0005 instead of 0.048).

## Known Issues

### Pause at beginning of audio
T3 may generate silence/filler tokens before actual speech. Investigate:
- T3 conditioning alignment
- How prompt_tokens influence generation start
- Whether text position embeddings are correct

### CPU Performance
T3 inference is very slow on CPU (~1 token/second for 520M params).
Use GPU for practical generation.
