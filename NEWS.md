# chatterbox 0.1.0 (development)

## Fidelity review vs chatterbox-tts 0.1.4 (June 2026)

Full top-to-bottom comparison against the Python reference; thanks to
@chris-english for the bug reports that prompted it (#1, #2, #5).

### Text front end
- `generate()` now applies `punc_norm()` unconditionally like the Python
  reference (whitespace collapse, first-letter capitalization,
  punctuation rewrites, trailing period). The missing trailing period
  was a major cause of missed end-of-speech (#1).
- Paralinguistic tokens (`[laughter]`, `[sigh]`, `[whisper]`, ...) now
  tokenize atomically instead of being spelled out letter by letter (#5).
- Fixed BPE corruption for inputs that fully merge to one token.

### Sampling
- Repetition penalty is sign-dependent (HF semantics) in all backends;
  the old divide-only form rewarded repeats with negative logits (#1).
- `top_p` defaults to 1.0 (disabled) like Python; `min_p` and
  `repetition_penalty` are now actually forwarded to the standard model.
- Degenerate-loop guard: the same token sampled 3x in a row stops
  generation with a warning and `eos_found = FALSE`.

### Conditioning
- Windowed-sinc resampler and Kaldi fbank ports (validated against
  torchaudio to < 1e-8); the speaker encoder now sees the features it
  was trained on.
- Reference audio capped at 10 s (S3Gen) / 6 s (tokenizer prompt), as
  upstream; voice encoder trims silence and uses Resemble's windowing.
- Prompt mel/token alignment fixed for references that are not a
  multiple of 40 ms.

### Other
- CFG unconditional branch, double-BOS prefill, exact GELU, fp32
  default (autocast now opt-in), CUDA/MPS availability fallback,
  batch-safe pad masks, Python-parity SOS/EOS token stripping.
- `conds.pt` no longer downloaded (unused by the R API).
