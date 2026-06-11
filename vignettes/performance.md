<!--
%\VignetteIndexEntry{Performance: Backends and GC Tuning}
%\VignetteEngine{simplermarkdown::mdweave_to_html}
%\VignetteEncoding{UTF-8}
-->
---
title: "Performance: Backends and GC Tuning"
---

# Performance

The chatterbox package runs Chatterbox TTS natively in R with three
inference backends. This vignette reports measured performance, explains
where the time actually goes, and documents the torch garbage-collection
settings that matter more than the backend choice.

## Test Configuration (June 2026)

- **GPU**: RTX 5060 Ti (16GB VRAM)
- **torch**: 0.17.0 (libtorch 2.8), float32 throughout
- **Test text**: ~130 speech tokens (~5s of audio), jfk.wav reference
- Scripts: `scripts/bench_backends.R`, `scripts/tune_gc.R`,
  `scripts/profile_backends.R`

**Scope caveat: all numbers are from this one machine.** The GC
mechanism is R/torch-side and applies on any CUDA GPU, and the cliff
is ratio arithmetic - collections strangle inference whenever the
model's reserved fraction of the card exceeds `reserved_rate`. So
severity scales with card size: a 24 GB card's ~19% floor sits under
the default 0.2 line and may never hit this at all, while small cards
live deep past it. Absolute ms/token figures and the cpp-vs-traced
ranking have not been verified on other hardware.

## Headline Numbers

Per generated token, warm:

| Backend | default torch GC settings | tuned GC settings |
|---------|--------------------------|-------------------|
| cpp (`backend = "cpp"`) | 229-429 ms | **19-28 ms** |
| traced (`traced = TRUE`) | 41-47 ms | **34-39 ms** |
| pure R | 920-1598 ms | **100-113 ms** |
| Python container (reference) | ~8 ms | - |

The single most important fact in this vignette: **with default torch
settings, inference is garbage-collection-bound, not compute-bound.**
Profiling (profvis + debrief) showed pure-R generation spending ~91% of
its wall time in R GC — and the cpp backend, despite running its whole
loop in C++, was throttled the same way (its allocations still flow
through the allocator that invokes R collections). Tune one option and
the backend ranking inverts: the compiled loop becomes the fastest
native path, ~3x off the Python container.

## The GC Story

R is the only thing that frees torch tensors: a GPU tensor's memory
returns when R finalizes its handle, during a collection. To avoid
running out of memory, torch's allocators *invoke full R collections
themselves*, governed by four options read **once, at torch startup**:

1. `torch.cuda_allocator_reserved_rate` (default 0.2): once torch has
   reserved more than this fraction of total VRAM, GPU allocations
   start invoking collections. The loaded model alone (~4.6 GB) is past
   20% of a 16 GB card, so every allocation can trigger one. **This is
   the speed knob.**
2. `torch.cuda_allocator_allocated_rate` (default 0.8): forces full
   collections once allocated memory crosses this fraction of the card.
   **This is the VRAM-ceiling knob** - it does not affect speed.
3. `torch.cuda_allocator_allocated_reserved_rate` (default 0.8): the
   fragmentation rule. No measurable effect in our sweeps; leave it.
4. `torch.threshold_call_gc` (default 4000): full `gc()` per N MB of
   cumulative host-side allocation. Raising it alone gave only ~1.5x,
   and it adds nothing once the reserved rate is set; leave it.

Because the settings are read at startup, they must be set **before
torch loads** - in `.Rprofile` or at the very top of a script. Setting
them after `library(torch)` (or any torch use) does nothing.

### Which knobs, at what ranges (one knob moved at a time)

| knob moved alone | speed | VRAM plateau |
|---|---|---|
| reserved_rate .2 -> .5 (others default) | 1113 -> 105 ms/tok (the whole win) | sets it |
| threshold_call_gc 4 GB -> 16/64 GB | 1113 -> ~700-800 (minor) | none |
| allocated_rate .8 -> .6 | none | caps lower (flat ~9.3 GB) |
| allocated_rate .8 -> .95 | none | climbs higher, no benefit |
| allocated_reserved_rate .6 / .95 | none | none |

The reserved rate is a cliff, not a dial: every value from 0.3 to 0.8
gave the same ~100-113 ms/token (pure R). All that matters is the
trigger line clearing what the loaded model reserves; past that, the
value only chooses how high the VRAM plateau sits (0.3 -> ~9 GB,
0.8 -> ~14 GB on a 16 GB card).

### Tuned settings

One option. `chatterbox_gc_options()` prints it for your card. For a
16 GB GPU:

```r
options(torch.cuda_allocator_reserved_rate = 0.5)
```

For a 6 GB GPU use `0.75` (the ~3.2 GB model floor is already 53% of a
small card, so the line must sit higher). Optionally add
`torch.cuda_allocator_allocated_rate = 0.6` to hold the VRAM plateau
lower at no speed cost - useful on shared GPUs.

**Rule of thumb: collect once per utterance, not thousands of times
inside it.** `tts_chunked()` calls `gc()` after each chunk; do the same
after each `generate()` in your own batch loops. That bounds garbage at
one utterance's worth with no measurable speed cost.

## Where the Remaining Time Goes

With GC tamed, both R-driven backends converge near ~110 ms/token of
shared loop cost: per-token tensor creation, KV-cache slice assignment,
and sampling, each crossing the R/C++ boundary. The traced graphs cut
the per-layer portion of that to ~35 ms/token total; the graphs
themselves are only ~30% of traced's per-token time.

Generation is inherently sequential (each token needs the previous
one), so per-token overhead multiplies by the full token count and
cannot be batched away.

## Backend Details

### cpp (`backend = "cpp"`) - fastest native, experimental

The T3 decode loop as a single `.Call()` into a compiled loop on the
ATen C++ API. With tuned GC settings: **19-28 ms/token**, no JIT
compilation cold start, and ~1 GB less VRAM than traced (no duplicate
weight copies). It also has no TorchScript dependency, so it is immune
to the deprecation that will eventually strand traced mode.

Under default GC settings it measured 230-430 ms/token, which is why it
was long believed slow: its allocations flow through the same allocator
that invokes R collections, even though the loop never returns to R.
Tune the one option and the compiled loop shows its true cost.

Still marked experimental: it shares traced's 350-position cache cap
(liftable - the loop attends only over valid positions, so a larger
cache costs VRAM but no speed), and has had less soak time than the
other paths. Requires libtorch headers at install time (auto-detected
by the configure script; without them it compiles to a stub).

### Traced (`traced = TRUE`)

`torch::jit_trace()` compiles the 30 T3 transformer layers and the CFM
estimator into TorchScript graphs.

- **Cold start**: ~50-60s one-time JIT compilation per session.
- **Warm**: ~35-39 ms/token (tuned GC).
- **Limits**: KV cache fixed at 350 positions (conditioning + ~190-270
  generated tokens); traced CFM fixed at 1024 mel frames - the prompt
  mel shares that budget, so a 10s reference leaves ~262 generated
  tokens before it falls back (gracefully, with a warning) to the
  eager estimator.
- **Memory**: ~1 GB above eager - traced graphs embed their own copies
  of the layer weights, and the eager originals stay loaded for the
  prefill pass and fallbacks.
- **Caveat**: TorchScript is deprecated upstream (maintenance mode);
  it works on current libtorch but has no future.

### Pure R (`backend = "r"`) - most capable

Everything eager. ~110 ms/token with tuned GC settings. No token cap
below the model's own limits, no extra VRAM, fully debuggable. The
right backend for long generations and for small cards.

## Memory

| component | size |
|-----------|------|
| model weights (fp32) | ~3.2 GB |
| traced graphs | ~1 GB extra |
| T3 KV cache | ~0.94 MB per position (CFG batch of 2) |
| flow/vocoder activations | transient, tens to hundreds of MB |

Reference audio barely matters: T3 sees it through a fixed 32-slot
perceiver, and the flow prompt adds ~50 mel frames per second of
(10s-capped) reference.

## When to Use What

- **Container** (`tts.api` backend): production speed (~8 ms/token).
- **cpp + tuned GC**: fastest native path (19-28 ms/token), no JIT
  warmup; utterance-length text (350-position cache cap).
- **Traced**: when libtorch headers were unavailable at install time
  but speed matters in a long-running session.
- **Pure R + tuned GC**: long texts, small cards, debugging, anywhere
  else.

## Optimizations Applied

- Fused scaled-dot-product attention throughout.
- Sign-dependent repetition penalty, vectorized (positive logits
  divided, negative multiplied - HF semantics) with no per-token
  GPU readbacks.
- CPU-first weight loading (halves peak VRAM during load).
- Fixed pre-generated CFM noise buffer (deterministic, no per-call
  RNG churn).
