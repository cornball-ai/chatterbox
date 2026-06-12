# chatterbox 0.1.0.2 (development)

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
- `backend = "cpp"` auto-sizes its KV cache, so generations of any
  length complete; with tuned GC settings, long-form native generation
  runs at container speed (0.30 vs 0.29 wall-seconds per audio-second).
- `generate()` gains `max_new_tokens` and `max_cache_len`.
- `tts_chunked()` actually enforces `chunk_size` now (it was dead
  code): run-on sentences split at comma boundaries.

## GC tuning and performance (June 2026)

- With torch's default allocator settings, inference is
  garbage-collection-bound: ~91% of pure-R generation wall time is R GC.
  One option fixes it: `torch.cuda_allocator_reserved_rate` set above
  the model's reserved fraction of the card (~10x pure-R speedup, ~15x
  for the cpp backend). New `chatterbox_gc_options()` prints the snippet
  for your GPU; the performance vignette has the full attribution table.
- `backend = "cpp"` is the fastest native path under tuned GC settings
  (19-28 ms/token on the test GPU, no JIT cold start). Still marked
  experimental. Its repetition penalty is now vectorized on-device.
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
