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

**Scope caveat: most numbers are from this one machine.** The GC
mechanism is R/torch-side and applies on any CUDA GPU, and the cliff
is ratio arithmetic - collections strangle inference whenever the
model's reserved fraction of the card exceeds `reserved_rate`. So
severity scales with card size: a 24 GB card's ~19% floor sits under
the default 0.2 line and may never hit this at all, while small cards
live deep past it. Absolute ms/token figures do not travel; the 6 GB
section below has the second measured machine, and
`chatterbox_defaults()` encodes both tiers.

## Headline Numbers

Per generated token, warm:

| Backend | default torch GC settings | tuned GC settings |
|---------|--------------------------|-------------------|
| jit (`backend = "jit"`) | - | **11 ms** (long-form) |
| cpp (retired; see below) | 229-429 ms | 9-28 ms |
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

Per-card recommendations (the rule: the trigger line, a fraction of
total VRAM, must clear what the loaded model reserves):

| card | reserved_rate | status |
|------|--------------|--------|
| 16 GB | 0.50 | measured (RTX 5060 Ti) |
| 12 GB | 0.50 | projected from the rule; not yet validated |
| 8 GB | 0.60 | projected from the rule; not yet validated |
| 6 GB | 0.75 | measured (GTX 1660 Ti) |

`chatterbox_defaults()` returns the full per-card setup (this option
plus backend and chunking thresholds) as a ready-to-paste snippet.

To validate on new hardware, run `scripts/tune_gc.R` with a few values;
the win is a cliff, so any rate that clears the floor gives full speed.
Optionally add `torch.cuda_allocator_allocated_rate = 0.6` to hold the
VRAM plateau lower at no speed cost - but only where 60% of the card
clears the model floor (roughly 8 GB cards and up); below that the cap
recreates the constant-collection regime.

### Measured on 6 GB hardware (GTX 1660 Ti)

| config (pure R unless noted) | ms/tok | VRAM |
|--------|--------|------|
| default | 1032-1811 | 3.6 GB flat |
| reserved_rate 0.75 | **300-360** | 4.4-5.4 GB oscillating |
| 0.75 + allocated_rate 0.6 | 423-441 (worse) | 4.7-4.9 GB |
| 0.75, jit, long-form (June 2026) | **35-38** | 4.7 GB |
| 0.75, traced (warm, short text) | 88-94 | 5.0 GB - tight, no OOM |
| 0.75, cpp (retired) | 150-163 | 4.4 GB stable |
| 0.75, long text (~20-23s audio) | 351-392, completes | 4.4 GB stable |
| 0.9 + 0.9 backstop, short text | 302-305, steadier | 5.3 GB flat |
| 0.9 + 0.9 backstop, long text | **OOM, both runs** | - |

The same mechanism on a different GPU generation: the untuned storm
reproduces, and one knob buys ~4x for pure R (not 10x - the card is
tight enough that the 0.8 backstop line still fires collections,
visible as the oscillating VRAM). The allocated_rate=0.6 row is the
floor rule above demonstrated: 60% of this card sits below the model
floor.

The jit row (validated June 2026, after the cpp retirement) settles the
backend question on small cards too: 35-38 ms/token long-form against
the container's 30 on the same box - within ~25% of Python, 2.6x faster
than traced, ~8x faster than pure R, in 4.7 GB. Traced is additionally
disqualified for long-form here: its 350-position cache cap truncates
this test's text at 120 tokens (~5 s of audio) without an EOS. The
earlier "traced wins on 6 GB" finding was an artifact of jit not yet
existing when those rows were measured. **backend = "jit" is the
recommendation on every measured card**; `chatterbox_defaults()`
returns it along with the GC tier.

The 0.9 rows are why the backstop stays at its default. Pushing both
lines to 90% runs steadier on short utterances (the card parks at a
flat 5.3 GB, container-style) but OOMs on long ones - and the autopsy
is instructive: allocated peaked at 4.88 GB, just UNDER the 5.04 GB
rescue line, while ~0.6 GB of fragmentation filled the rest of the
card. R frees tensors only at collection (unlike Python's refcounting,
which is how the container lives at 5.5 GB safely), so the gap between
the backstop and the top of the card must absorb fragmentation plus
the largest single allocation. 0.75 with default backstops completes
20+ second generations on this card; that is the recommendation.

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

### jit (`backend = "jit"`) - fastest native

Each token's full 30-layer forward runs as one TorchScript function,
compiled once per session (in milliseconds) via `torch::jit_compile()`.
**11 ms/token long-form** with tuned GC - container-parity territory -
with an auto-sized KV cache so generation always completes (~1 MB VRAM
per position; pass `max_cache_len` to bound it). No shape freezing, no
per-size recompilation, no JIT-trace cold start, and no duplicate
weight copies (the script borrows the model's tensors by reference).

It replaced an equivalent C++ backend (9 ms/token long-form, 19-28 ms
short - within ~20%) that required a configure script and linking
against the torch package's private libtorch: that linkage broke on
install order, was permanently dead in any CRAN-built binary, and
could go stale on torch upgrades. The TorchScript route shares traced
mode's deprecation caveat but none of those failure modes. The 6 GB
rows below labelled cpp are historical measurements of the retired
backend; jit was validated on that hardware in June 2026 (35-38
ms/token - see the 6 GB section).

### Where the speed actually comes from

Same long text, T3 stage, tuned GC (`scripts/bench_jit_vs_eager.R`):

| variant | ms/token |
|---------|----------|
| pure R, nn_module forward path | 87-88 |
| lean eager R: identical math, ATen builtins only (no nn_module/nnf dispatch) | 71 |
| jit (one TorchScript call per token) | 11 |

Rewriting eager R in the leanest possible style buys ~20%: the
dominant cost is the per-op R-to-lantern call itself (~190 us across
~370 ops/token), not wrapper style. The 8x comes from removing the
per-op R call structurally, which is what the TorchScript step does.

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

- **jit + tuned GC**: fastest native path (~11 ms/token), no warmup,
  any length (auto-sized cache). The default choice on a GPU.
- **Container** (`tts.api` backend): still the fastest overall
  (~8 ms/token) and the most battle-tested.
- **Traced**: long-running sessions where the ~50s trace cost amortizes
  and utterance-length text fits its 350-position cache.
- **Pure R + tuned GC**: debugging, CPU-only, anywhere else.

## Optimizations Applied

- Fused scaled-dot-product attention throughout.
- Sign-dependent repetition penalty, vectorized (positive logits
  divided, negative multiplied - HF semantics) with no per-token
  GPU readbacks.
- CPU-first weight loading (halves peak VRAM during load).
- Fixed pre-generated CFM noise buffer (deterministic, no per-call
  RNG churn).
