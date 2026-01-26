#!/usr/bin/env python3
"""Check what attention mask the estimator uses internally."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get forward source
print("\n=== Estimator forward source ===")
try:
    src = inspect.getsource(type(estimator).forward)
    # Print just the mask-related lines
    for line in src.split('\n'):
        if 'mask' in line.lower() or 'attn' in line.lower():
            print(line)
except Exception as e:
    print(f"Error: {e}")

# Test with explicit attention_mask=None vs attention_mask=0
print("\n=== Testing attention mask differences ===")
torch.manual_seed(42)
with torch.no_grad():
    batch = 2
    time = 50

    x = torch.randn(batch, 80, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    mu = torch.randn(batch, 80, time, device='cuda')
    t = torch.ones(batch, device='cuda') * 0.5
    spks = torch.randn(batch, 80, device='cuda')
    cond = torch.randn(batch, 80, time, device='cuda')

    # Standard forward (uses internal mask logic)
    out1 = estimator(x, mask, mu, t, spks, cond)
    print(f"Standard forward: mean={out1.mean().item():.6f}, std={out1.std().item():.6f}")

# Check static_chunk_size
print(f"\nestimator.static_chunk_size: {estimator.static_chunk_size}")
