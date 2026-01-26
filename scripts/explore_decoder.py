#!/usr/bin/env python3
"""Explore CFM decoder architecture."""

import torch
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
decoder = model.s3gen.flow.decoder
estimator = decoder.estimator

print("\n=== Decoder (CausalConditionalCFM) ===")
print(f"Type: {type(decoder).__name__}")
for name, child in decoder.named_children():
    print(f"  {name}: {type(child).__name__}")

print("\n=== Estimator (ConditionalDecoder) ===")
print(f"Type: {type(estimator).__name__}")
for name, child in estimator.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in list(child.named_children())[:3]:
            print(f"    {n2}: {type(c2).__name__}")

print("\n=== Down blocks structure ===")
for i, (resnet, transformer_blocks, downsample) in enumerate(estimator.down_blocks):
    print(f"down_block[{i}]:")
    print(f"  resnet: {type(resnet).__name__}")
    print(f"  transformer_blocks: {len(transformer_blocks)} blocks")
    if len(transformer_blocks) > 0:
        tb = transformer_blocks[0]
        print(f"    transformer_block[0]:")
        for name, child in tb.named_children():
            print(f"      {name}: {type(child).__name__}")
    print(f"  downsample: {type(downsample).__name__}")

print("\n=== Mid blocks structure ===")
for i, (resnet, transformer_blocks) in enumerate(estimator.mid_blocks):
    print(f"mid_block[{i}]:")
    print(f"  resnet: {type(resnet).__name__}")
    print(f"  transformer_blocks: {len(transformer_blocks)} blocks")

print("\n=== Up blocks structure ===")
for i, (resnet, transformer_blocks, upsample) in enumerate(estimator.up_blocks):
    print(f"up_block[{i}]:")
    print(f"  resnet: {type(resnet).__name__}")
    print(f"  transformer_blocks: {len(transformer_blocks)} blocks")
    print(f"  upsample: {type(upsample).__name__}")

print("\n=== Final layers ===")
print(f"final_block: {type(estimator.final_block).__name__}")
print(f"final_proj: {type(estimator.final_proj).__name__}")

# Check key dimensions
print("\n=== Key dimensions ===")
if hasattr(estimator, 'static_chunk_size'):
    print(f"static_chunk_size: {estimator.static_chunk_size}")

# Get resnet input/output channels
resnet0 = estimator.down_blocks[0][0]
print(f"\nResnetBlock structure:")
for name, child in resnet0.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'in_channels'):
        print(f"    in={child.in_channels}, out={child.out_channels}")

# Transformer block structure
tb = estimator.down_blocks[0][1][0]
print(f"\nTransformerBlock structure:")
for name, child in tb.named_children():
    print(f"  {name}: {type(child).__name__}")
