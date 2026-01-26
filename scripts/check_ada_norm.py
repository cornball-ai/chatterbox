#!/usr/bin/env python3
"""Check AdaLayerNorm usage."""

import torch
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

down_block = estimator.down_blocks[0]
transformer = down_block[1][0]

print(f"\nuse_ada_layer_norm: {transformer.use_ada_layer_norm}")
print(f"use_ada_layer_norm_zero: {transformer.use_ada_layer_norm_zero}")

print(f"\nnorm1 type: {type(transformer.norm1).__name__}")
print(f"norm3 type: {type(transformer.norm3).__name__}")

# Check norm1 attributes if it's AdaLayerNorm
if hasattr(transformer.norm1, 'linear'):
    print(f"\nnorm1.linear: {transformer.norm1.linear}")
    print(f"norm1.emb type: {type(transformer.norm1.emb).__name__}")

# Test norm1 forward with timestep
import torch
x = torch.randn(2, 50, 256).cuda()
t = torch.randn(2, 1024).cuda()

try:
    y = transformer.norm1(x, t)
    print(f"\nnorm1(x, t): shape={y.shape}, mean={y.mean().item():.6f}")
except Exception as e:
    print(f"\nnorm1(x, t) error: {e}")

try:
    y = transformer.norm1(x)
    print(f"norm1(x): shape={y.shape}, mean={y.mean().item():.6f}")
except Exception as e:
    print(f"norm1(x) error: {e}")
