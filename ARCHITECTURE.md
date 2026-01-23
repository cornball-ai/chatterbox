# Chatterbox Architecture

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

## Validation Status

Sorted by **max difference** (largest first = most suspect):

| # | Component | Status | Max Diff | Test Script |
|---|-----------|--------|----------|-------------|
| 8 | CFM Decoder | ✅ | 0.052 | `scripts/test_cfm_full.R` |
| 9 | HiFi-GAN Vocoder | ✅ | 0.026 | `scripts/test_hifigan.R` |
| 5 | CAMPPlus | ✅ | 0.0015 | `scripts/test_campplus.R` |
| 7 | Conformer Encoder | ✅ | 0.0004 | `scripts/test_encoder_steps.R` |
| 2 | Voice Encoder | ✅ | 0.00026 | `scripts/test_voice_encoder.R` |
| 4 | T3 Model | ✅ | 0.00003 | `scripts/test_t3_llama.R` |
| 4 | T3 Conditioning | ✅ | 0.000002 | `scripts/test_t3_cond.R` |
| 6 | Mel Extractor | ✅ | 0.000001 | `scripts/test_voice_encoder.R` |
| 3 | S3 Tokenizer | ✅ | 0 (exact) | `scripts/test_s3tokenizer.R` |
| 1 | Text Tokenizer | ✅ | 0 (exact) | (uses same tokenizer.json) |

## Key Test Scripts

### Quick validation (run these first)
```bash
# Component with largest diff - CFM decoder
Rscript scripts/test_cfm_full.R

# Second largest - vocoder
Rscript scripts/test_hifigan.R

# End-to-end with Python reference
Rscript scripts/test_s3gen_vs_python.R
```

### By component
```
1. Text Tokenizer     - (exact match, uses same JSON)
2. Voice Encoder      - scripts/test_voice_encoder.R
3. S3 Tokenizer       - scripts/test_s3tokenizer.R
4. T3 Model           - scripts/test_t3_llama.R, scripts/test_t3_cond.R
5. CAMPPlus           - scripts/test_campplus.R
6. Mel Extractor      - scripts/compare_mel.R
7. Conformer Encoder  - scripts/test_encoder_steps.R, scripts/test_conformer_full.R
8. CFM Decoder        - scripts/test_cfm_full.R, scripts/test_cfm_estimator.R
9. HiFi-GAN Vocoder   - scripts/test_hifigan.R
```

## Known Issues

### Pause at beginning of audio
T3 may generate silence/filler tokens before actual speech. Investigate:
- T3 conditioning alignment
- How prompt_tokens influence generation start
- Whether text position embeddings are correct

### CPU Performance
T3 inference is very slow on CPU (~1 token/second for 520M params).
Use GPU for practical generation.
