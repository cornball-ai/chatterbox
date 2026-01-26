#!/usr/bin/env python3
"""Save UpsampleConformerEncoder reference outputs for R validation."""

import torch
from safetensors.torch import save_file
from chatterbox import ChatterboxTTS

# Load model
print("Loading ChatterboxTTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device)

# Get the flow module from s3gen
flow = tts.s3gen.flow
flow.eval()
print(f"Flow module type: {type(flow)}")

# Get encoder
encoder = flow.encoder
print(f"Encoder type: {type(encoder)}")

# Create test input (speech token embeddings)
print("\nCreating test input...")
batch_size = 1
seq_len = 50  # ~2 seconds of speech tokens (25 tokens/sec)

# Use actual token embedding for realistic test
torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (batch_size, seq_len), device=device)
test_input = flow.input_embedding(test_tokens)
test_lens = torch.tensor([seq_len], device=device)

print(f"Test input shape: {test_input.shape}")
print(f"Test input mean: {test_input.mean().item():.6f}, std: {test_input.std().item():.6f}")

# Run through encoder with hooks to capture intermediates
intermediates = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().clone()
        else:
            intermediates[name] = output.detach().clone()
    return hook

# Register hooks
hooks = []
hooks.append(encoder.embed.register_forward_hook(make_hook('after_embed')))
hooks.append(encoder.pre_lookahead_layer.register_forward_hook(make_hook('after_pre_lookahead')))
hooks.append(encoder.encoders[0].register_forward_hook(make_hook('after_encoder0')))
hooks.append(encoder.encoders[-1].register_forward_hook(make_hook('after_encoders')))
hooks.append(encoder.up_layer.register_forward_hook(make_hook('after_upsample')))
hooks.append(encoder.up_embed.register_forward_hook(make_hook('after_up_embed')))
hooks.append(encoder.up_encoders[-1].register_forward_hook(make_hook('after_up_encoders')))
hooks.append(encoder.after_norm.register_forward_hook(make_hook('after_final_norm')))

# Run forward pass
print("\nRunning encoder forward pass...")
with torch.no_grad():
    full_output, output_masks = encoder(test_input, test_lens)
    print(f"Full output shape: {full_output.shape}")
    print(f"Output masks shape: {output_masks.shape}")

# Remove hooks
for h in hooks:
    h.remove()

# Print intermediate shapes
print("\nIntermediate shapes:")
for name, tensor in intermediates.items():
    print(f"  {name}: {tensor.shape}")

# Save outputs
print("\nSaving outputs...")
outputs = {
    "test_input": test_input.cpu().float().contiguous(),
    "test_lens": test_lens.cpu().contiguous(),
    "full_output": full_output.cpu().float().contiguous(),
}

# Add intermediates
for name, tensor in intermediates.items():
    outputs[name] = tensor.cpu().float().contiguous()

save_file(outputs, "/outputs/conformer_reference.safetensors")
print("Saved to /outputs/conformer_reference.safetensors")

# Print statistics
print("\n=== Statistics ===")
print(f"Input - mean: {test_input.mean().item():.6f}, std: {test_input.std().item():.6f}")
for name in ['after_embed', 'after_pre_lookahead', 'after_encoders', 'after_upsample', 'after_up_encoders']:
    if name in intermediates:
        t = intermediates[name]
        print(f"{name} - mean: {t.mean().item():.6f}, std: {t.std().item():.6f}")
print(f"Final output - mean: {full_output.mean().item():.6f}, std: {full_output.std().item():.6f}")
