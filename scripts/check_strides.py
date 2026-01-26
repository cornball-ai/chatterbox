#!/usr/bin/env python3
"""Check downsample and upsample strides."""

from chatterbox.tts import ChatterboxTTS
import torch

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

down_block = estimator.down_blocks[0]
down_conv = down_block[2]
print(f"\ndown_conv type: {type(down_conv).__name__}")
for name, mod in down_conv.named_modules():
    if hasattr(mod, 'stride'):
        print(f"  {name}.stride: {mod.stride}")
        print(f"  {name}.kernel_size: {mod.kernel_size}")

up_block = estimator.up_blocks[0]
up_conv = up_block[2]
print(f"\nup_conv type: {type(up_conv).__name__}")
for name, mod in up_conv.named_modules():
    if hasattr(mod, 'stride'):
        print(f"  {name}.stride: {mod.stride}")
        print(f"  {name}.kernel_size: {mod.kernel_size}")

# Test actual behavior
print("\n=== Testing actual behavior ===")
x = torch.randn(1, 256, 50).cuda()
mask = torch.ones(1, 1, 50).cuda()

y_down = down_conv(x * mask)
print(f"Input: {x.shape} -> down_conv -> {y_down.shape}")

# For up_conv, we need to match dimensions
y_up = up_conv(x * mask)
print(f"Input: {x.shape} -> up_conv -> {y_up.shape}")
