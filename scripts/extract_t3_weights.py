#!/usr/bin/env python3
"""Extract T3 model weights for R comparison."""

import torch
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")

t3 = model.t3
print(f"T3 Model: {type(t3).__name__}")

# Extract state dict
state_dict = {}
for name, param in t3.named_parameters():
    state_dict[name] = param.cpu().contiguous()
    print(f"  {name}: {param.shape}")

for name, buffer in t3.named_buffers():
    state_dict[name] = buffer.cpu().contiguous()
    print(f"  {name} (buffer): {buffer.shape}")

print(f"\nTotal: {len(state_dict)} tensors")

save_file(state_dict, "/outputs/t3_weights.safetensors")
print("Saved to /outputs/t3_weights.safetensors")
