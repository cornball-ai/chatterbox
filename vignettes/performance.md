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

## Headline Numbers

Per generated token, warm:

| Backend | default torch GC settings | tuned GC settings |
|---------|--------------------------|-------------------|
| pure R | 920-1598 ms | **107-113 ms** |
| traced (`traced = TRUE`) | 41-47 ms | **34-37 ms** |
| cpp (`backend = "cpp"`, experimental) | 229-386 ms | not separately tuned |
| Python container (reference) | ~8 ms | - |

The single most important fact in this vignette: **with default torch
settings, pure-R inference spends ~91% of its wall time in R garbage
collection** (measured with profvis + debrief). The backend comparison
is almost a footnote to the GC story.

## The GC Story

R is the only thing that frees torch tensors: a GPU tensor's memory
returns when R finalizes its handle, during a collection. To avoid
running out of memory, torch's allocators *invoke full R collections
themselves*, governed by four options read **once, at torch startup**:

1. `torch.threshold_call_gc` (default 4000): full `gc()` per N MB of
   cumulative host-side allocation. Every tensor handle ticks this
   odometer; a per-token loop crosses 4 GB constantly.
2. `torch.cuda_allocator_reserved_rate` (default 0.2): once torch has
   reserved more than this fraction of total VRAM, GPU allocations
   start invoking collections. The loaded model alone (~4.5 GB) is past
   20% of a 16 GB card, so every allocation can trigger one.
3. `torch.cuda_allocator_allocated_rate` (default 0.8) and
4. `torch.cuda_allocator_allocated_reserved_rate` (default 0.8): the
   backstop - force full collections near the top of the card. Leave
   both at their defaults; they are the self-regulation that prevents
   out-of-memory, not the problem.

Because the settings are read at startup, they must be set **before
torch loads** - in `.Rprofile` or at the very top of a script. Setting
them after `library(torch)` (or any torch use) does nothing.

### Tuned settings

`chatterbox_gc_options()` prints the snippet for your card. For a 16 GB
GPU:

```r
options(
    torch.cuda_allocator_reserved_rate = 0.5,
    torch.threshold_call_gc = 16000
)
```

For a 6 GB GPU use `reserved_rate = 0.75`: the trigger line is a
fraction of the card, and the ~3.2 GB model floor already exceeds 50%
of a small card.

### Measured trade (5 consecutive generations per config)

| config | pure-R ms/tok | traced ms/tok | VRAM trajectory |
|--------|--------------|---------------|-----------------|
| default (.2 / 4 GB) | 920-1598 | 41-47 | flat 4.6-4.9 GB |
| rate .5, cpu 16 GB | 107-113 | 34-37 | 9.3 -> 13.8 GB |
| rate .7, cpu 64 GB | 112-113 | 34-36 | 12.2 -> 14.9 GB |
| rate .9 + gc() per generation | 109-110 | 34-40 | plateaus 15.4 GB |

The speed win saturates at moderate settings; pushing further only
raises the VRAM plateau. The creep is homeostatic (the 0.8 backstop
forces collections near the top of the card), but the steady state
lives high.

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

### Traced (`traced = TRUE`) - fastest

`torch::jit_trace()` compiles the 30 T3 transformer layers and the CFM
estimator into TorchScript graphs.

- **Cold start**: ~50-60s one-time JIT compilation per session.
- **Warm**: ~35 ms/token.
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

### cpp (`backend = "cpp"`) - experimental

The T3 decode loop as a single `.Call()` into a compiled loop on the
ATen C++ API. Kept as a hedge: it does not depend on TorchScript, so it
survives the deprecation that will eventually strand traced mode.
Currently ~230-390 ms/token - slower than traced, faster than untuned
pure R. Requires libtorch headers at install time (auto-detected by the
configure script; without them it compiles to a stub). Shares traced's
350-position cache cap.

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
- **Traced**: long-running R sessions, utterance-length text.
- **Pure R + tuned GC**: long texts, small cards, debugging, anywhere
  Docker isn't.
- **cpp**: experimental; revisit when TorchScript removal gets real.

## Optimizations Applied

- Fused scaled-dot-product attention throughout.
- Sign-dependent repetition penalty, vectorized (positive logits
  divided, negative multiplied - HF semantics) with no per-token
  GPU readbacks.
- CPU-first weight loading (halves peak VRAM during load).
- Fixed pre-generated CFM noise buffer (deterministic, no per-call
  RNG churn).
