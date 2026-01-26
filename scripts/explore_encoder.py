#!/usr/bin/env python3
"""Explore UpsampleConformerEncoder architecture."""

import torch
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
encoder = model.s3gen.flow.encoder

print("\n=== Encoder (UpsampleConformerEncoder) ===")
print(f"Type: {type(encoder).__name__}")
for name, child in encoder.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, '__len__'):
        print(f"    length: {len(child)}")

print("\n=== Encoder attributes ===")
for attr in ['output_size', 'embed_dim', 'attention_heads', 'linear_units',
             'num_blocks', 'dropout_rate', 'positional_dropout_rate',
             'attention_dropout_rate', 'input_layer', 'normalize_before',
             'static_chunk_size', 'use_dynamic_chunk', 'use_dynamic_left_chunk',
             'causal']:
    if hasattr(encoder, attr):
        print(f"  {attr}: {getattr(encoder, attr)}")

print("\n=== Input layer structure ===")
for name, child in encoder.embed.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'in_channels'):
        print(f"    in={child.in_channels}, out={child.out_channels}")
    if hasattr(child, 'in_features'):
        print(f"    in={child.in_features}, out={child.out_features}")

print("\n=== Conformer blocks ===")
print(f"Number of blocks: {len(encoder.encoders)}")
if len(encoder.encoders) > 0:
    block = encoder.encoders[0]
    print(f"\nBlock 0 structure ({type(block).__name__}):")
    for name, child in block.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, 'named_children'):
            for n2, c2 in list(child.named_children())[:5]:
                print(f"    {n2}: {type(c2).__name__}")

print("\n=== Self-attention structure ===")
if len(encoder.encoders) > 0:
    attn = encoder.encoders[0].self_attn
    print(f"Type: {type(attn).__name__}")
    for name, child in attn.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, 'in_features'):
            print(f"    in={child.in_features}, out={child.out_features}")

print("\n=== Convolution module structure ===")
if len(encoder.encoders) > 0:
    conv = encoder.encoders[0].conv_module
    print(f"Type: {type(conv).__name__}")
    for name, child in conv.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, 'in_channels'):
            print(f"    in={child.in_channels}, out={child.out_channels}, kernel={child.kernel_size}")

print("\n=== Upsample structure ===")
upsample = encoder.upsample
print(f"Type: {type(upsample).__name__}")
for name, child in upsample.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'in_channels'):
        print(f"    in={child.in_channels}, out={child.out_channels}")
    if hasattr(child, 'in_features'):
        print(f"    in={child.in_features}, out={child.out_features}")

print("\n=== Weight key patterns ===")
# Get weight keys
state_dict = model.s3gen.state_dict()
encoder_keys = [k for k in state_dict.keys() if k.startswith('flow.encoder.')]
print(f"Total encoder keys: {len(encoder_keys)}")
print("\nSample keys:")
for k in sorted(encoder_keys)[:30]:
    print(f"  {k}: {list(state_dict[k].shape)}")

# Check for pos encoding
pos_keys = [k for k in encoder_keys if 'pos' in k.lower()]
print(f"\nPositional encoding keys ({len(pos_keys)}):")
for k in pos_keys:
    print(f"  {k}: {list(state_dict[k].shape)}")
