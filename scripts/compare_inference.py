#!/usr/bin/env python3
"""Compare Python T3 inference details"""

import sys
sys.path.insert(0, '/app/.venv/lib/python3.11/site-packages')

import torch
from chatterbox.tts import ChatterboxTTS
import torchaudio
from safetensors.torch import save_file
import inspect

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")

text = "cornball AI is doing something for our country!"
ref_path = "/scripts/reference.wav"

print(f"Text: {text}")

# Check T3.inference signature and source
print("\n=== T3.inference signature ===")
sig = inspect.signature(model.t3.inference)
for name, param in sig.parameters.items():
    if param.default is not inspect.Parameter.empty:
        print(f"  {name}: {param.default}")

# Generate with DEFAULT parameters (as model.generate would use)
print("\n=== Generating with defaults (like model.generate) ===")
wav_default = model.generate(text, audio_prompt_path=ref_path)
print(f"Default duration: {wav_default.shape[-1] / 24000:.2f}s")

# Generate with the same parameters as R
print("\n=== Generating with R parameters ===")
wav_r_params = model.generate(
    text,
    audio_prompt_path=ref_path,
    temperature=0.8,
    top_p=0.95,  # R uses 0.95, Python default is 1.0
    min_p=0.05,
    repetition_penalty=1.2,
    cfg_weight=0.5
)
print(f"R-params duration: {wav_r_params.shape[-1] / 24000:.2f}s")

# Save both
torchaudio.save("/outputs/python_default.wav", wav_default.cpu(), 24000)
torchaudio.save("/outputs/python_r_params.wav", wav_r_params.cpu(), 24000)
print("\nSaved both audio files")

# Also trace token generation
print("\n=== Tracing T3 token generation ===")
cond = model.prepare_conditionals(ref_path)
text_tokens = model.t3.tokenize_text([text])
t3_cond = model.t3.prepare_cond(**model.conds.t3_cond.__dict__)

with torch.inference_mode():
    # Use defaults
    tokens_default = model.t3.inference(
        cond=t3_cond,
        text_tokens=text_tokens,
        max_new_tokens=1000
    )
print(f"Default tokens: {tokens_default.shape[1]}")
print(f"First 30: {tokens_default[0, :30].tolist()}")

with torch.inference_mode():
    # Use R params
    tokens_r = model.t3.inference(
        cond=t3_cond,
        text_tokens=text_tokens,
        max_new_tokens=1000,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.2,
        cfg_weight=0.5
    )
print(f"R-params tokens: {tokens_r.shape[1]}")
print(f"First 30: {tokens_r[0, :30].tolist()}")
