#!/usr/bin/env python3
"""Trace BasicTransformerBlock structure in detail."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get a transformer block
tfm = estimator.down_blocks[0][1][0]
print(f"\n=== BasicTransformerBlock ===")
print(f"Type: {type(tfm).__name__}")

# Print all children
print("\nChildren:")
for name, child in tfm.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2).__name__}")
            if hasattr(c2, 'weight') and c2.weight is not None:
                print(f"      weight: {c2.weight.shape}")

# Check attn1 (self-attention)
print("\n=== Attention (attn1) ===")
attn = tfm.attn1
print(f"Type: {type(attn).__name__}")
print(f"heads: {attn.heads}")
print(f"inner_dim: {attn.inner_dim if hasattr(attn, 'inner_dim') else 'N/A'}")
for name, child in attn.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'weight') and child.weight is not None:
        print(f"    weight: {child.weight.shape}")

# Check ff (feed-forward)
print("\n=== FeedForward (ff) ===")
ff = tfm.ff
print(f"Type: {type(ff).__name__}")
for name, child in ff.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2).__name__}")
            if hasattr(c2, 'weight') and c2.weight is not None:
                print(f"      weight: {c2.weight.shape}")

# Get forward source
print("\n=== Forward Source ===")
try:
    src = inspect.getsource(type(tfm).forward)
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check CausalResnetBlock1D
print("\n\n=== CausalResnetBlock1D ===")
resnet = estimator.down_blocks[0][0]
print(f"Type: {type(resnet).__name__}")

for name, child in resnet.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2).__name__}")
            if hasattr(c2, 'weight') and c2.weight is not None:
                print(f"      weight: {c2.weight.shape}")

try:
    src = inspect.getsource(type(resnet).forward)
    print("\nForward source:")
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check CausalBlock1D
print("\n\n=== CausalBlock1D ===")
causal_block = resnet.block1
print(f"Type: {type(causal_block).__name__}")
for name, child in causal_block.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2).__name__}")
            if hasattr(c2, 'weight') and c2.weight is not None:
                print(f"      weight: {c2.weight.shape}")

try:
    src = inspect.getsource(type(causal_block).forward)
    print("\nForward source:")
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check CausalConv1d
print("\n\n=== CausalConv1d ===")
causal_conv = causal_block.block[0]
print(f"Type: {type(causal_conv).__name__}")
print(f"weight: {causal_conv.weight.shape}")
if hasattr(causal_conv, 'padding'):
    print(f"padding: {causal_conv.padding}")

try:
    src = inspect.getsource(type(causal_conv).forward)
    print("\nForward source:")
    print(src)
except Exception as e:
    print(f"Error: {e}")

# Check downsample
print("\n\n=== Downsample ===")
downsample = estimator.down_blocks[0][2]
print(f"Type: {type(downsample).__name__}")
print(f"weight: {downsample.weight.shape}")

try:
    src = inspect.getsource(type(downsample).forward)
    print("\nForward source:")
    print(src)
except Exception as e:
    print(f"Error: {e}")
