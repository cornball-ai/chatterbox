#!/usr/bin/env python3
"""Trace CFM decoder (CausalConditionalCFM) forward pass."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
decoder = model.s3gen.flow.decoder

# Get source code
print("\n=== CausalConditionalCFM forward source ===")
try:
    src = inspect.getsource(type(decoder).forward)
    print(src[:3000])
except Exception as e:
    print(f"Error: {e}")

# Check attributes
print("\n=== Decoder attributes ===")
for name in dir(decoder):
    if not name.startswith('_'):
        obj = getattr(decoder, name)
        if not callable(obj):
            print(f"  {name}: {obj}")

# Check estimator
print("\n=== Estimator type ===")
print(f"decoder.estimator: {type(decoder.estimator).__name__}")

# Get solve_euler source
print("\n=== solve_euler source ===")
if hasattr(decoder, 'solve_euler'):
    try:
        src = inspect.getsource(decoder.solve_euler)
        print(src[:3000])
    except Exception as e:
        print(f"Error: {e}")

# Test forward with debugging
print("\n=== Testing forward with debug ===")
torch.manual_seed(42)
with torch.no_grad():
    batch = 1
    time = 50

    mu = torch.randn(batch, 80, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    spks = torch.randn(batch, 80, device='cuda')
    cond = torch.randn(batch, 80, time, device='cuda')

    print(f"mu: shape={mu.shape}, mean={mu.mean().item():.6f}")
    print(f"spks: shape={spks.shape}, mean={spks.mean().item():.6f}")
    print(f"cond: shape={cond.shape}, mean={cond.mean().item():.6f}")

    # Check if decoder has inference_cfg_rate
    if hasattr(decoder, 'inference_cfg_rate'):
        print(f"inference_cfg_rate: {decoder.inference_cfg_rate}")

    result, _ = decoder(mu=mu, mask=mask, spks=spks, cond=cond, n_timesteps=10)
    print(f"Output: shape={result.shape}, mean={result.mean().item():.6f}, std={result.std().item():.6f}")
