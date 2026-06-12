# chatterbox 0.1.0.9 (development)

- New `chatterbox_defaults()`: detects the GPU and returns the full
  recommended setup (GC options, backend, token budget, chunking
  threshold) as a pasteable snippet.
- 6GB hardware validation: jit measures 35-38 ms/token vs container 30;
  per-card guidance updated (jit is the fastest native backend on every
  measured card).

# chatterbox 0.1.0.8 (development)

- New `generate_batch()`: several texts, one batched S3Gen synthesis
  pass; padded rows validated to match single runs (mel diff <= 0.005).
- `s3gen$inference()` accepts ragged batches via `speech_token_lens`.

# chatterbox 0.1.0.7 (development)

- New `voice_convert()`: speech-to-speech voice conversion (port of
  Python ChatterboxVC); re-renders source speech in a target voice,
  preserving the source timing.

# chatterbox 0.1.0.6 (development)

- `generate(skip_vocoder = TRUE)` returns the mel spectrogram instead of
  audio (Python 0.1.7 parity).
- New `save_voice_embedding()`/`load_voice_embedding()`: torch_save-based
  voice presets, reusable across sessions without the reference audio.

# chatterbox 0.1.0.5 (development)

- New `integrated_loudness()` and `normalize_loudness()` (ITU-R
  BS.1770-4, pure base R, matches pyloudnorm to 6 decimals);
  `create_voice_embedding()` gains `norm_loudness`, defaulting to TRUE
  for turbo models (Python parity).
- `read_audio()` downmixes stereo files by channel mean (librosa
  parity); previously the right channel was silently dropped.
- Parity reference retargeted to chatterbox-tts 0.1.7.

# chatterbox 0.1.0.4 (development)

- `chatterbox_gc_options()` now returns a classed list of the
  recommended `options()` values (apply with `do.call(options, ...)`
  before torch loads); the printed advice moved to its print method.

# chatterbox 0.1.0.3 (development)

## C++ apparatus retired in favor of a TorchScript backend (June 2026)

- New `backend = "jit"`: each token's 30-layer forward runs as one
  TorchScript function (`torch::jit_compile`, compiled per session in
  milliseconds). 11 ms/token long-form with tuned GC settings, within
  ~20% of the C++ backend it replaces, auto-sized KV cache, no
  compiled code.
- Deleted `src/`, `configure`, and `cleanup`: the C++ backend linked
  against the torch package's private libtorch, which broke on install
  order, was dead in CRAN-built binaries, and could go stale on torch
  upgrades. chatterbox is now a pure-R package.
- Measured dispatch attribution (see the performance vignette): even
  eager R written directly against ATen builtins keeps a ~70 ms/token
  floor; the per-op R call is the cost, not wrapper style.

## Container parity for long-form (June 2026)

- The CFM estimator's attention uses the fused SDPA kernel: the mel
  stage runs 2.5x faster and stops triggering GC storms at long
  sequence lengths.
- The fast backend auto-sizes its KV cache, so generations of any
  length complete; with tuned GC settings, long-form native generation
  runs at container speed (0.30 vs 0.29 wall-seconds per audio-second).
  (Measured on the C++ backend, since replaced by `backend = "jit"`,
  which inherits the auto-sized cache.)
- `generate()` gains `max_new_tokens` and `max_cache_len`.
- `tts_chunked()` actually enforces `chunk_size` now (it was dead
  code): run-on sentences split at comma boundaries.

## GC tuning and performance (June 2026)

- With torch's default allocator settings, inference is
  garbage-collection-bound: ~91% of pure-R generation wall time is R GC.
  One option fixes it: `torch.cuda_allocator_reserved_rate` set above
  the model's reserved fraction of the card (~10x pure-R speedup, ~15x
  for the compiled-loop backend). New `chatterbox_gc_options()` prints the snippet
  for your GPU; the performance vignette has the full attribution table.
- The compiled-loop backend measured fastest native under tuned GC
  (19-28 ms/token short-form; that C++ backend has since been replaced
  by `backend = "jit"` at ~11 ms/token long-form). Repetition penalty
  vectorized on-device.
- `tts_chunked()` collects garbage once per chunk, bounding dead tensor
  handles (and VRAM creep) at one utterance's worth.
- Performance vignette rewritten around these findings, with a
  hardware-scope caveat: numbers are from one GPU; the mechanism
  generalizes, the magnitudes may not.

# chatterbox 0.1.0.1 (development)

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
- Degenerate-loop guard: the same token sampled 10x in a row stops
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
