#!/usr/bin/env python3
"""Explore encoder weight keys and structures."""

import torch
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
encoder = model.s3gen.flow.encoder

# Get weight keys
state_dict = model.s3gen.state_dict()
encoder_keys = [k for k in state_dict.keys() if k.startswith('flow.encoder.')]

print(f"=== All encoder keys ({len(encoder_keys)}) ===")
for k in sorted(encoder_keys):
    shape = list(state_dict[k].shape)
    print(f"  {k}: {shape}")

print("\n=== Pre-lookahead layer ===")
pll = encoder.pre_lookahead_layer
print(f"Type: {type(pll).__name__}")
for name, child in pll.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'in_channels'):
        print(f"    in={child.in_channels}, out={child.out_channels}, kernel={child.kernel_size}, stride={child.stride}")

print("\n=== Positional encoding ===")
pos_enc = encoder.embed.pos_enc
print(f"Type: {type(pos_enc).__name__}")
for name, child in pos_enc.named_children():
    print(f"  {name}: {type(child).__name__}")

# Check if pe is learnable
if hasattr(pos_enc, 'pe'):
    print(f"  pe shape: {pos_enc.pe.shape if hasattr(pos_enc.pe, 'shape') else 'N/A'}")
    print(f"  pe is parameter: {isinstance(pos_enc.pe, torch.nn.Parameter)}")

print("\n=== Upsample1D layer ===")
up = encoder.up_layer
print(f"Type: {type(up).__name__}")
for name, child in up.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'in_channels'):
        print(f"    in={child.in_channels}, out={child.out_channels}, kernel={child.kernel_size}, stride={child.stride}, padding={child.padding}")

# Check forward signature
import inspect
print(f"\nUpsample1D.forward signature: {inspect.signature(up.forward)}")
