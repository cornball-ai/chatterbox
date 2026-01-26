#!/usr/bin/env python3
"""Trace the embed layer in detail."""

import torch
from chatterbox import ChatterboxTTS

# Load model
print("Loading model...")
device = "cuda"
tts = ChatterboxTTS.from_pretrained(device)
encoder = tts.s3gen.flow.encoder
encoder.eval()

# Create test input (same as reference)
torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (1, 50), device=device)
test_input = tts.s3gen.flow.input_embedding(test_tokens)
test_lens = torch.tensor([50], device=device)

print(f"Input shape: {test_input.shape}")
print(f"Input mean: {test_input.mean().item():.6f}, std: {test_input.std().item():.6f}")

# Trace embed
print("\n=== Tracing embed layer ===")
with torch.no_grad():
    # Get mask - create manually
    T = test_input.size(1)
    # mask where True = valid position
    masks = torch.ones(1, 1, T, dtype=torch.bool, device=device)
    print(f"Masks shape: {masks.shape}")

    # Check embed.out structure
    print(f"embed.out type: {type(encoder.embed.out)}")
    print(f"embed.out children: {list(encoder.embed.out._modules.keys())}")

    # Step through embed.out
    x = test_input
    for i, (name, module) in enumerate(encoder.embed.out._modules.items()):
        x = module(x)
        print(f"After {name} ({type(module).__name__}): mean={x.mean().item():.6f}, std={x.std().item():.6f}")

    # Now positional encoding
    print(f"\nembed.pos_enc type: {type(encoder.embed.pos_enc)}")
    result = encoder.embed.pos_enc(x)
    if isinstance(result, tuple):
        xs, pos_emb = result
        print(f"After pos_enc: xs mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
        print(f"pos_emb shape: {pos_emb.shape}")
    else:
        print(f"Result type: {type(result)}")

    # Full embed forward
    print("\n=== Full embed.forward ===")
    full_result = encoder.embed(test_input, masks)
    xs_full = full_result[0]
    print(f"Full embed output: mean={xs_full.mean().item():.6f}, std={xs_full.std().item():.6f}")
