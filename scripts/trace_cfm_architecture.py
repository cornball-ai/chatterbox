#!/usr/bin/env python3
"""Trace CFM estimator architecture in detail."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen
flow = s3gen.flow
decoder = flow.decoder

print("\n=== CausalConditionalCFM ===")
print(f"Type: {type(decoder).__name__}")
print(f"sigma: {decoder.sigma if hasattr(decoder, 'sigma') else 'N/A'}")

print("\nDecoder attributes:")
for attr in dir(decoder):
    if not attr.startswith('_'):
        obj = getattr(decoder, attr)
        if hasattr(obj, 'weight') or hasattr(obj, 'named_children'):
            print(f"  {attr}: {type(obj).__name__}")

print("\n=== ConditionalDecoder (estimator) ===")
estimator = decoder.estimator
print(f"Type: {type(estimator).__name__}")

# Print full structure
def print_module_tree(module, prefix="", depth=0):
    if depth > 3:
        return
    for name, child in module.named_children():
        print(f"{prefix}{name}: {type(child).__name__}")
        if hasattr(child, 'weight'):
            if hasattr(child.weight, 'shape'):
                print(f"{prefix}  weight: {child.weight.shape}")
        if hasattr(child, 'bias') and child.bias is not None:
            if hasattr(child.bias, 'shape'):
                print(f"{prefix}  bias: {child.bias.shape}")
        print_module_tree(child, prefix + "  ", depth + 1)

print("\nFull module tree:")
print_module_tree(estimator, "  ")

# Get forward source
print("\n=== Estimator forward source ===")
try:
    src = inspect.getsource(type(estimator).forward)
    print(src[:2000])
except Exception as e:
    print(f"Error: {e}")

# Check actual shapes by running forward
print("\n=== Test forward shapes ===")
with torch.no_grad():
    batch = 1
    time = 100
    out_channels = 80  # mel bins

    # Inputs based on ConditionalDecoder forward signature
    x = torch.randn(batch, out_channels, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    mu = torch.randn(batch, out_channels, time, device='cuda')
    t = torch.ones(batch, device='cuda') * 0.5
    spks = torch.randn(batch, out_channels, device='cuda')
    cond = torch.randn(batch, out_channels, time, device='cuda')

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  mu: {mu.shape}")
    print(f"  t: {t.shape}")
    print(f"  spks: {spks.shape}")
    print(f"  cond: {cond.shape}")

    # Forward through estimator
    output = estimator(x, mask, mu, t, spks, cond)
    print(f"\nOutput shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in estimator.parameters())
print(f"\nTotal estimator parameters: {total_params:,}")
