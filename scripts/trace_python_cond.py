#!/usr/bin/env python3
"""Trace Python conditioning structure for comparison with R"""

import sys
sys.path.insert(0, '/app/.venv/lib/python3.11/site-packages')

import torch
from chatterbox.tts import ChatterboxTTS
from safetensors.torch import save_file

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")

text = "cornball AI is doing something for our country!"
ref_path = "/scripts/reference.wav"

print(f"\nText: {text}")

# Prepare conditionals
print("\n=== Conditioning ===")
model.prepare_conditionals(ref_path)
print(f"model.conds type: {type(model.conds)}")

# Get t3 cond
t3_cond = model.conds.t3
print(f"t3_cond type: {type(t3_cond)}")
print(f"t3_cond attributes: {[a for a in dir(t3_cond) if not a.startswith('_')]}")

# Look at t3_cond values
if hasattr(t3_cond, '__dict__'):
    for k, v in t3_cond.__dict__.items():
        if hasattr(v, 'shape'):
            print(f"  t3_cond.{k}: {v.shape}")
        else:
            print(f"  t3_cond.{k}: {v}")

# Generate with tracing
print("\n=== Running generate with verbose output ===")
# Monkey-patch to trace
original_inference = model.t3.inference

def traced_inference(cond, text_tokens, **kwargs):
    print(f"\n[TRACE] T3.inference called:")
    print(f"  text_tokens shape: {text_tokens.shape}")
    print(f"  text_tokens: {text_tokens[0].tolist()}")
    print(f"  cond type: {type(cond)}")
    if hasattr(cond, '__dict__'):
        for k, v in cond.__dict__.items():
            if hasattr(v, 'shape'):
                print(f"  cond.{k}: {v.shape}")
            else:
                print(f"  cond.{k}: {v}")
    result = original_inference(cond, text_tokens, **kwargs)
    print(f"  output tokens: {result.shape}")
    print(f"  first 30: {result[0, :30].tolist()}")
    return result

model.t3.inference = traced_inference

wav = model.generate(text, audio_prompt_path=ref_path)
print(f"\nFinal audio duration: {wav.shape[-1] / 24000:.2f}s")
