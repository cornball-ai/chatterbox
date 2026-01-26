#!/usr/bin/env python3
"""Trace encoder forward - simplified."""

import torch
from safetensors.torch import save_file
from chatterbox import ChatterboxTTS

# Load model
print("Loading model...")
tts = ChatterboxTTS.from_pretrained('cuda')
encoder = tts.s3gen.flow.encoder
encoder.eval()

# Create test input
torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (1, 50), device='cuda')
test_input = tts.s3gen.flow.input_embedding(test_tokens)
test_lens = torch.tensor([50], device='cuda')

print(f"Input: {test_input.shape}, mean={test_input.mean().item():.6f}")

# Use the encoder's forward method with hooks to capture intermediates
intermediates = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().cpu().float().contiguous()
        else:
            intermediates[name] = output.detach().cpu().float().contiguous()
    return hook

# Register hooks
hooks = []
hooks.append(encoder.embed.register_forward_hook(make_hook("embed")))
hooks.append(encoder.pre_lookahead_layer.register_forward_hook(make_hook("pre_lookahead")))

for i, enc in enumerate(encoder.encoders):
    hooks.append(enc.register_forward_hook(make_hook(f"encoder_{i}")))

hooks.append(encoder.up_layer.register_forward_hook(make_hook("up_layer")))
hooks.append(encoder.up_embed.register_forward_hook(make_hook("up_embed")))

for i, enc in enumerate(encoder.up_encoders):
    hooks.append(enc.register_forward_hook(make_hook(f"up_encoder_{i}")))

hooks.append(encoder.after_norm.register_forward_hook(make_hook("after_norm")))

# Run forward
with torch.no_grad():
    output, output_lens = encoder(test_input, test_lens)

# Print and save
print("\nIntermediate values:")
for name in sorted(intermediates.keys()):
    tensor = intermediates[name]
    print(f"  {name}: {list(tensor.shape)}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")

intermediates["input"] = test_input.cpu().float().contiguous()
intermediates["output"] = output.cpu().float().contiguous()

save_file(intermediates, "/outputs/encoder_steps.safetensors")
print("\nSaved to /outputs/encoder_steps.safetensors")

# Clean up hooks
for h in hooks:
    h.remove()
