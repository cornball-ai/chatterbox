#!/usr/bin/env python3
"""Check BasicTransformerBlock structure."""

import torch
import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get down block transformer
down_block = estimator.down_blocks[0]
transformer_blocks = down_block[1]
transformer = transformer_blocks[0]

print(f"\ntransformer type: {type(transformer).__name__}")
print(f"\ntransformer forward signature:")
print(inspect.signature(transformer.forward))

print(f"\ntransformer attributes:")
for name, module in transformer.named_children():
    print(f"  {name}: {type(module).__name__}")

# Check forward source
print(f"\n=== BasicTransformerBlock forward source ===")
src = inspect.getsource(type(transformer).forward)
for line in src.split('\n')[:40]:
    print(line)
