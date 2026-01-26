#!/usr/bin/env python3
"""Trace positional encoding."""

import torch
from chatterbox import ChatterboxTTS

# Load model
print("Loading model...")
tts = ChatterboxTTS.from_pretrained('cuda')
pos_enc = tts.s3gen.flow.encoder.embed.pos_enc
pos_enc.eval()

print(f"pos_enc type: {type(pos_enc)}")
print(f"pos_enc class name: {pos_enc.__class__.__name__}")

# Check for xscale attribute
if hasattr(pos_enc, 'xscale'):
    print(f"pos_enc.xscale: {pos_enc.xscale}")

# Create test input
x = torch.randn(1, 50, 512).cuda()
x = x * 0.007  # Scale to match layernorm output

print(f"\nInput x: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

# Run forward
with torch.no_grad():
    result = pos_enc(x)
    if isinstance(result, tuple):
        xs, pos_emb = result
        print(f"Output xs: mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
        print(f"pos_emb shape: {pos_emb.shape}")
        print(f"pos_emb std: {pos_emb.std().item():.6f}")

        # Check if xs == x or xs == x + something
        diff = (xs - x).abs().max().item()
        print(f"Max diff between xs and x: {diff:.6f}")

        if diff > 0.001:
            print("=> xs was modified (positional embedding was added)")
        else:
            print("=> xs was NOT modified")
    else:
        print(f"Unexpected result type: {type(result)}")

# Check the PE tensor
if hasattr(pos_enc, 'pe'):
    pe = pos_enc.pe
    print(f"\npe buffer shape: {pe.shape}")
    print(f"pe std: {pe.std().item():.6f}")
