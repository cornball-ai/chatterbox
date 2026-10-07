# chatterbox

Pure R port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
on torch. No Python, no compiled code: the C++ decode backend was removed in
PR #7 and `backend = "jit"` (TorchScript via `torch::jit_compile()`) replaced
it. Component internals, the validation record against the Python reference,
and the resolved port bugs live in `ARCHITECTURE.md`; read it before touching
a model file.

## Python reference

Target: chatterbox-tts 0.1.7 (PyPI). Container `chatterbox-tts:0.1.7-blackwell`
(PyPI 0.1.7 + transformers 5.2.0 + torch 2.7.0 cu128, no API server). The
fp32 English path is numerically identical to 0.1.4. Mount the host HF cache
so weights are not downloaded again:

```bash
docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD/outputs:/outputs" \
  chatterbox-tts:0.1.7-blackwell python /outputs/your_script.py
```

## Architecture

```
chatterbox
├── Text Tokenizer (BPE)
├── Voice Encoder (speaker embeddings)
├── T3 Model (text → speech tokens)
│   ├── Llama backbone (GPT-2 for Turbo)
│   ├── Perceiver resampler
│   └── Attention with KV cache
└── S3Gen (speech tokens → waveform)
    ├── CAMPPlus speaker encoder, Conformer encoder, CFM decoder
    └── HiFi-GAN vocoder
```

## Source map

| File | Holds |
|------|-------|
| `R/tts.R` | `generate()`, `generate_batch()`, `tts_chunked()`, `tts_to_file()`, `quick_tts()`, text normalization |
| `R/t3.R` | T3 config, pure-R decode loop, shared sampler `.sample_speech_token` |
| `R/t3_jit.R`, `R/t3_turbo_jit.R` | TorchScript decode step per architecture, session cache, `.get_layer_weights` |
| `R/llama.R`, `R/llama_traced.R` | Llama 520M blocks (RoPE, RMSNorm, SwiGLU); jit_trace path |
| `R/s3gen.R`, `R/conformer.R`, `R/hifigan.R` | S3Gen: flow matching, Conformer encoder, HiFT vocoder |
| `R/s3tokenizer.R`, `R/speaker_encoder.R`, `R/kaldi_fbank.R` | S3 tokenizer (FSQ), CAMPPlus, Kaldi fbank |
| `R/voice_encoder.R`, `R/voice_io.R`, `R/vc.R` | Speaker embedding, save/load voices, `voice_convert()` |
| `R/safetensors.R`, `R/tokenizer.R`, `R/download.R` | Pure R safetensors reader (F16/BF16 by hand, empty keys), BPE tokenizer, hfhub downloads |
| `R/resident.R`, `R/serve.R` | Resident model held across calls; HTTP server (`serve()`), see `inst/chatterbox.service` |
| `R/audio_utils.R`, `R/resample.R`, `R/loudness.R` | Audio I/O, resampling, BS.1770-4 loudness |
| `R/gc_options.R` | `chatterbox_gc_options()` |

## Usage

```r
library(chatterbox)
model <- load_chatterbox(chatterbox("cuda"))          # downloads weights once
result <- generate(model, "Hello world!", "reference_voice.wav", backend = "jit")
write_audio(result$audio, result$sample_rate, "output.wav")
quick_tts("Hello!", "ref.wav", "out.wav")             # one-liner, loads per call
```

`tts_chunked()` splits long text by sentence and collects garbage per chunk.
`load_chatterbox_turbo()` loads the Turbo model (fewer FLOPs, smaller VRAM).

## Model weights

hfhub puts them in `~/.cache/huggingface/hub/`. Standard: `ResembleAI/chatterbox`
(`t3_cfg.safetensors` 2.1 GB, `s3gen.safetensors` 1.1 GB, `ve.safetensors`,
`tokenizer.json`). Turbo: `ResembleAI/chatterbox-turbo`. `conds.pt` is not
downloaded on purpose: a nested torch pickle R cannot read, so the R API
always needs a reference voice. A partial download shows up as header offsets
past the file size; `download_chatterbox_models(force = TRUE)` fixes it.

## R torch pitfalls that shaped this code

The pytorch-migration skill has the general list. The ones this port hit:

- `self$m(x)` and `self$m$forward(x)` both work (torch 0.17.0). The code uses
  `$forward()` throughout; keep it for consistency.
- `torch_sort()` indices and `nn_embedding` inputs are 1-indexed; speech
  tokens are kept 0-indexed everywhere else, and the EOS check converts.
- Tensor methods return new tensors: `tokens <- tokens$sub(1L)`, never a bare
  `tokens$sub(1L)`. A dropped assignment here once produced 19 s of silence.
- `torch_clamp()` can change dtype; cast back to long before embedding.
- `x * 0.5` promotes float16 to float32; use `x$mul(0.5)`.
- `conv_transpose1d` with kernel < stride needs `padding = 0` and
  `output_padding = stride - kernel`.
- TorchScript under this lantern: no `torch.nn.functional`, no dtype
  constants; use ATen builtins (`torch.matmul`, `torch.silu`,
  `torch.scaled_dot_product_attention`, `.float()`), wrap R ints with
  `torch::jit_scalar()`. In-place writes to passed tensors persist across
  the boundary.

## Performance (RTX 5060 Ti 16 GB, torch 0.17 / libtorch 2.8, June 2026)

`vignettes/performance.md` has the full story. Two facts dominate:

1. GC settings matter more than backend. With torch defaults ~91% of pure-R
   wall time was R GC. Set `options(torch.cuda_allocator_reserved_rate = 0.5)`
   before torch loads (0.75 on 6 GB, 0.6 on 8 GB); `chatterbox_gc_options()`
   prints the snippet. Collect once per utterance in batch loops.
2. `backend = "jit"` is the fastest native path.

| backend | ms/token | notes |
|---------|----------|-------|
| container (Python) | ~8 | reference |
| `backend = "jit"` | 11 | one TorchScript function per session, no warmup |
| `traced = TRUE` | 34-39 | ~50 s trace per session; 350-position cache cap |
| pure R | 87-88 | no caps, fully debuggable |

Use jit with tuned GC on any GPU; the container for production via
tts.api/gpu.ctl; traced for long sessions of short utterances; pure R for
debugging and CPU.

## Development

```bash
r -e 'tinyrox::document(); tinypkgr::install()'
r -e 'tinytest::test_package("chatterbox")'
bash tools/check_tarball.sh      # before every CRAN build, on the real tree
```

`R CMD build` packages the working directory, so untracked scratch in the
package root ships. CI checks out a clean tree and cannot see it; the local
`check_tarball.sh` run is the only check that does. Inspect the tarball too:
`tar tzf chatterbox_<version>.tar.gz | awk -F/ 'NF>1 {print $2}' | sort -u`.

## Related

- tts.api uses this package as `backend = "native"`; the container backend
  is the alternative when Docker is available.
- pytorch-migration skill for porting patterns.
