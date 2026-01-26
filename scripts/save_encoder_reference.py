#!/usr/bin/env python3
"""Save complete encoder forward pass for validation."""

import torch
from safetensors.torch import save_file
from chatterbox import ChatterboxTTS

# Load model
print("Loading model...")
tts = ChatterboxTTS.from_pretrained('cuda')
encoder = tts.s3gen.flow.encoder
encoder.eval()

# Create test input (same as reference)
torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (1, 50), device='cuda')
test_input = tts.s3gen.flow.input_embedding(test_tokens)
test_lens = torch.tensor([50], device='cuda')

print(f"Input shape: {test_input.shape}")
print(f"Input mean: {test_input.mean().item():.6f}, std: {test_input.std().item():.6f}")

# Run encoder forward
with torch.no_grad():
    output, output_lens = encoder(test_input, test_lens)
    print(f"\nEncoder output:")
    print(f"  shape: {output.shape}")
    print(f"  mean: {output.mean().item():.6f}")
    print(f"  std: {output.std().item():.6f}")
    print(f"  output_lens: {output_lens.tolist()}")

# Save
outputs = {
    "input": test_input.cpu().float().contiguous(),
    "input_lens": test_lens.cpu().int().contiguous(),
    "output": output.cpu().float().contiguous(),
    "output_lens": output_lens.cpu().int().contiguous(),
}

save_file(outputs, "/outputs/encoder_reference.safetensors")
print("\nSaved to /outputs/encoder_reference.safetensors")
