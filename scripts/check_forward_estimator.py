#!/usr/bin/env python3
"""Check forward_estimator method."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
decoder = model.s3gen.flow.decoder

# Get forward_estimator source
print("\n=== forward_estimator source ===")
try:
    src = inspect.getsource(decoder.forward_estimator)
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check if there's any preprocessing
print("\n=== Testing forward_estimator vs estimator ===")
torch.manual_seed(42)
with torch.no_grad():
    x = torch.randn(2, 80, 50, device='cuda')
    mask = torch.ones(2, 1, 50, device='cuda')
    mu = torch.randn(2, 80, 50, device='cuda')
    t = torch.ones(2, device='cuda') * 0.5
    spks = torch.randn(2, 80, device='cuda')
    cond = torch.randn(2, 80, 50, device='cuda')

    # Direct estimator call
    out1 = decoder.estimator(x, mask, mu, t, spks, cond)
    print(f"estimator output: mean={out1.mean().item():.6f}, std={out1.std().item():.6f}")

    # forward_estimator call
    out2 = decoder.forward_estimator(x, mask, mu, t, spks, cond)
    print(f"forward_estimator output: mean={out2.mean().item():.6f}, std={out2.std().item():.6f}")

    print(f"Difference: {(out1 - out2).abs().max().item():.6f}")
