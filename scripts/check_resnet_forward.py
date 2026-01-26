#!/usr/bin/env python3
"""Check the CausalResnetBlock1D forward signature."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get down block resnet
down_block = estimator.down_blocks[0]
down_resnet = down_block[0]

print(f"\ndown_resnet type: {type(down_resnet).__name__}")
print(f"\ndown_resnet forward signature:")
print(inspect.signature(down_resnet.forward))

print(f"\ndown_resnet attributes:")
for name, module in down_resnet.named_children():
    print(f"  {name}: {type(module).__name__}")

# Check the estimator forward source
print(f"\n=== Estimator forward source (excerpt) ===")
src = inspect.getsource(type(estimator).forward)
# Print first 50 lines
for i, line in enumerate(src.split('\n')[:60]):
    print(f"{i+1:3}: {line}")
