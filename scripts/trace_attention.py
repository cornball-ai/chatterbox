#!/usr/bin/env python3
"""Trace Attention module structure."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get attention from transformer block
tfm = estimator.down_blocks[0][1][0]
attn = tfm.attn1

print(f"\n=== Attention ===")
print(f"Type: {type(attn).__name__}")

# Print all attributes
for name in dir(attn):
    if not name.startswith('_'):
        obj = getattr(attn, name)
        if not callable(obj):
            print(f"  {name}: {obj}")

print(f"\nchildren:")
for name, child in attn.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'weight') and child.weight is not None:
        print(f"    weight: {child.weight.shape}")
    if hasattr(child, 'bias') and child.bias is not None:
        print(f"    bias: {child.bias.shape}")

# Get forward source
print("\n=== Attention Forward Source ===")
try:
    src = inspect.getsource(type(attn).forward)
    print(src[:3000])
except Exception as e:
    print(f"Error: {e}")

# Check SinusoidalPosEmb
print("\n\n=== SinusoidalPosEmb ===")
sin_emb = estimator.time_embeddings
print(f"Type: {type(sin_emb).__name__}")
for name in dir(sin_emb):
    if not name.startswith('_'):
        obj = getattr(sin_emb, name)
        if not callable(obj):
            print(f"  {name}: {obj}")

try:
    src = inspect.getsource(type(sin_emb).forward)
    print("\nForward source:")
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check TimestepEmbedding (time_mlp)
print("\n\n=== TimestepEmbedding ===")
time_mlp = estimator.time_mlp
print(f"Type: {type(time_mlp).__name__}")
for name, child in time_mlp.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'weight') and child.weight is not None:
        print(f"    weight: {child.weight.shape}")

try:
    src = inspect.getsource(type(time_mlp).forward)
    print("\nForward source:")
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check what causal_padding is for CausalConv1d
print("\n\n=== CausalConv1d details ===")
causal_conv = estimator.down_blocks[0][0].block1.block[0]
print(f"causal_padding: {causal_conv.causal_padding}")
print(f"kernel_size: {causal_conv.kernel_size}")
print(f"stride: {causal_conv.stride}")
print(f"padding: {causal_conv.padding}")
print(f"dilation: {causal_conv.dilation}")

# Check downsample
downsample = estimator.down_blocks[0][2]
print(f"\nDownsample:")
print(f"  causal_padding: {downsample.causal_padding}")
print(f"  kernel_size: {downsample.kernel_size}")
print(f"  stride: {downsample.stride}")

# Check upsample
upsample = estimator.up_blocks[0][2]
print(f"\nUpsample:")
print(f"  causal_padding: {upsample.causal_padding}")
print(f"  kernel_size: {upsample.kernel_size}")
print(f"  stride: {upsample.stride}")

# Test downsample behavior
with torch.no_grad():
    x = torch.randn(1, 256, 50, device='cuda')
    y = downsample(x)
    print(f"\nDownsample test: {x.shape} -> {y.shape}")

    y2 = upsample(y)
    print(f"Upsample test: {y.shape} -> {y2.shape}")
