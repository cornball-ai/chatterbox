#!/usr/bin/env python3
"""List estimator attributes."""

import torch
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

print("\n=== Estimator attributes ===")
for name, module in estimator.named_children():
    print(f"{name}: {type(module).__name__}")

# Check time_emb specifically
if hasattr(estimator, 'time_emb'):
    print(f"\ntime_emb structure:")
    for n, m in estimator.time_emb.named_modules():
        print(f"  {n}: {type(m).__name__}")
