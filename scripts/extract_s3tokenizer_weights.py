#!/usr/bin/env python3
"""Extract S3 Tokenizer weights for R comparison."""

import torch
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")

s3_tokenizer = model.s3gen.tokenizer
print(f"S3 Tokenizer: {type(s3_tokenizer).__name__}")

# Extract state dict
state_dict = {}
for name, param in s3_tokenizer.named_parameters():
    state_dict[f"tokenizer.{name}"] = param.cpu().contiguous()
    print(f"  {name}: {param.shape}")

for name, buffer in s3_tokenizer.named_buffers():
    state_dict[f"tokenizer.{name}"] = buffer.cpu().contiguous()
    print(f"  {name} (buffer): {buffer.shape}")

print(f"\nTotal: {len(state_dict)} tensors")

save_file(state_dict, "/outputs/s3tokenizer_weights.safetensors")
print("Saved to /outputs/s3tokenizer_weights.safetensors")
