# chatterbox 0.2.0.1 (development)

- `serve()` resolves a voice-library name (e.g. `"Barry"`) against
  `voices_dir` before treating the `voice` field as a path. Previously a
  like-named file or directory in the server's working directory shadowed
  the library voice, so `/v1/audio/speech` returned a 500 ("cannot open
  the connection"). A path is now accepted only when it is a regular file.

# chatterbox 0.2.0

First CRAN release. Gathers the 0.1.0.1 - 0.1.0.16 development series:
a complete pure-R port of Chatterbox TTS (no Python, no compiled code),
voice cloning, long-form chunked synthesis, an OpenAI-compatible
`serve()`, a TorchScript (`jit`) decode backend at container speed, and
automatic CUDA GC tuning. Per-change detail for the series is below.

# chatterbox 0.1.0.16 (development)

- `chatterbox()` gains a `tune_gc` argument (default TRUE) to opt out of
  the CUDA GC tuning added in 0.1.0.15. The tuning is a deliberate,
  persistent `options()` side effect (torch reads the allocator rates
  later, at CUDA init), documented in `?chatterbox`; pass
  `tune_gc = FALSE` to skip it. No behavior change at the default.

# chatterbox 0.1.0.15 (development)

- `chatterbox()` now tunes torch's CUDA garbage-collection rates before the
  first CUDA op. torch reads `torch.cuda_allocator_reserved_rate` (and
  `torch.threshold_call_gc`) once at lazy CUDA init; the 0.2 default floor
  meant gc ran on nearly every allocation once a model occupied more than
  20% of VRAM, which was 53% of inference wall time. The floor is now the
  model's footprint as a fraction of VRAM (4.1GB regular, 3.6GB turbo): e.g. a
  16GB card gets 0.26 / 0.23, a 6GB card 0.68 / 0.60. `threshold_call_gc` is
  raised to 16000 MB. All set ahead of `cuda_is_available()`. Turbo is ~2x
  faster on a 16GB card (10.7s -> 5.3s for a 16s utterance). An explicit
  user-set option still wins. See torch's memory-management vignette.

# chatterbox 0.1.0.14 (development)

- `read_audio()` now detects the audio container from the file's magic
  bytes (RIFF/WAVE, ID3, MP3 frame sync) instead of trusting the
  extension. A reference saved as PCM/WAV but named `.mp3` (or vice
  versa) previously ran the wrong decoder and produced NaN garbage,
  silently corrupting voice cloning; it now decodes correctly.

# chatterbox 0.1.0.13 (development)

- `serve()` now caches each voice embedding (by reference path + mtime)
  and reuses it across requests, instead of re-encoding the reference on
  every `/v1/audio/speech` call. Per-request re-encoding churned voice
  GPU tensors and raced the CUDA caching allocator, intermittently
  producing NaN speaker conditioning - seen as a "missing value where
  TRUE/FALSE needed" 500 and as degraded voice cloning (~33-50% of
  requests on both an RTX 5060 Ti and a GTX 1660 Ti; 0 with the cache).
  `trim_silence()` now raises a clear error instead of the cryptic one if
  NaN audio ever reaches it.

# chatterbox 0.1.0.12 (development)

- `serve()` now uses the `jit` backend for turbo as well as standard (was
  eager `"r"` for turbo, written before the turbo jit decode step
  existed). A turbo serve now runs the fast GPT-2 jit decode (~8x faster
  per token).

# chatterbox 0.1.0.11 (development)

- Turbo's GPT-2 tokenizer now emits the paralinguistic/emotion tags
  (`[sigh]`, `[laugh]`, `[whispering]`, `[cough]`, ...) as single special
  tokens. `load_gpt2_tokenizer()` builds an added-token split-list and
  `tokenize_text_gpt2()` splits on it before BPE; previously the tags were
  byte-BPE'd into `[`, `sigh`, `]` and never rendered.

# chatterbox 0.1.0.10 (development)

- New `t3_inference_turbo_jit()`: a TorchScript decode step for turbo's
  GPT-2 backbone, selected by `generate(turbo, backend = "jit")`. ~8x
  faster per token than the eager turbo path (the turbo counterpart of
  `t3_inference_jit`).
- Fixed turbo correctness (it was producing nonsense): the HF GPT-2
  Conv1D projection weights are now transposed for the `nn_linear`
  reimplementation (non-square ones were failing to load -> random
  weights), and `gpt2_model$forward` now adds the `wpe` absolute position
  embeddings that HF `GPT2Model` applies. With jit, turbo is ~1.6x faster
  than the standard model at comparable VRAM.

# chatterbox 0.1.0.9 (development)

- `chatterbox()` now constructs *and* loads the model by default (one
  call, like Python `from_pretrained`). Pass `load = FALSE` for the bare
  object. **Mildly breaking**: code that used `chatterbox()` as a cheap
  constructor before a separate `load_chatterbox()` now needs
  `load = FALSE` (or relies on `load_chatterbox()` being idempotent).
- `load_chatterbox()` / `load_chatterbox_turbo()` are idempotent: an
  already-loaded model is returned unchanged.
- `generate(output_path = )` also writes the audio to a WAV and adds a
  `path` element; `tts_to_file()` is now a thin wrapper over it.
- `generate()` defaults `normalize_text = FALSE`. The internal-caps
  mitigation patched a since-fixed (column-major/STFT) bug and was
  flattening intended emphasis; punctuation normalization still always
  runs. `normalize_tts_text(caps =, punctuation =)` is the single entry.
- `generate()` now errors clearly when the input exceeds the T3
  text-token limit instead of crashing, and sizes the traced CFM from the
  actual generated token count (no text-length guessing).
- `tts_chunked()` is the long-form layer: word-safe splitting, voice
  resolved once, and T3 run first so batching and the per-card memory cap
  use actual speech-token lengths.
- `serve()` routes synthesis through `tts_chunked()` (long-text splitting
  + per-card batching) and forwards more request knobs.

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
