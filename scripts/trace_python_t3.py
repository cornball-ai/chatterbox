#!/usr/bin/env python3
"""Trace Python T3 inference for comparison"""

import sys
sys.path.insert(0, '/app/.venv/lib/python3.11/site-packages')

import torch
import inspect
from chatterbox.tts import ChatterboxTTS
import torchaudio
from safetensors.torch import save_file

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")

# Check method signatures
print("\n=== Method signatures ===")
print(f"prepare_conditionals: {inspect.signature(model.prepare_conditionals)}")
print(f"generate: {inspect.signature(model.generate)}")

# Look at source of prepare_conditionals
print("\n=== prepare_conditionals source ===")
try:
    print(inspect.getsource(model.prepare_conditionals))
except:
    print("Could not get source")

# Use same text
text = "cornball AI is doing something for our country!"
print(f"\nText: {text}")

# Reference
ref_path = "/scripts/reference.wav"

# Check if reference exists
import os
print(f"Reference exists: {os.path.exists(ref_path)}")

# Load reference audio manually
wav, sr = torchaudio.load(ref_path)
print(f"Loaded reference: {wav.shape}, sr={sr}")

# Just use generate directly and trace
print("\n=== Generate with verbose tracing ===")
wav_out = model.generate(
    text,
    audio_prompt_path=ref_path,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.2,
    cfg_weight=0.5
)
print(f"Output: {wav_out.shape}")
print(f"Duration: {wav_out.shape[-1] / 24000:.2f}s")

torchaudio.save("/outputs/python_cornball.wav", wav_out.cpu(), 24000)
print("Saved to /outputs/python_cornball.wav")
