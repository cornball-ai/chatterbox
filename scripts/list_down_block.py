#!/usr/bin/env python3
"""List down_block attributes."""

import torch
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

print("\n=== down_blocks[0] attributes ===")
down_block = estimator.down_blocks[0]
for name, module in down_block.named_children():
    print(f"{name}: {type(module).__name__}")

print("\n=== mid_blocks[0] attributes ===")
mid_block = estimator.mid_blocks[0]
for name, module in mid_block.named_children():
    print(f"{name}: {type(module).__name__}")

print("\n=== input_blocks ===")
print(f"Number of input_blocks: {len(estimator.down_blocks)}")

# Check if there's an input_blocks
if hasattr(estimator, 'input_blocks'):
    print(f"input_blocks exists: {type(estimator.input_blocks)}")
