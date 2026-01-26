#!/usr/bin/env python3
"""Check up_block structure."""

from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

print("\n=== up_blocks structure ===")
print(f"Number of up_blocks: {len(estimator.up_blocks)}")

up_block = estimator.up_blocks[0]
print(f"\nup_block type: {type(up_block).__name__}")

for i, (name, module) in enumerate(up_block.named_children()):
    print(f"  {i}: {name} -> {type(module).__name__}")
