#!/usr/bin/env python3
"""Verify positional encoding behavior."""

import torch
from chatterbox import ChatterboxTTS

# Load model
print("Loading model...")
tts = ChatterboxTTS.from_pretrained('cuda')
pos_enc = tts.s3gen.flow.encoder.embed.pos_enc
pos_enc.eval()

xscale = pos_enc.xscale
print(f"xscale: {xscale}")

# Create test input (small std like after layernorm)
torch.manual_seed(42)
x = torch.randn(1, 50, 512).cuda() * 0.007

print(f"\nInput x: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
print(f"x * xscale: std={x.std().item() * xscale:.6f}")

# Run forward
with torch.no_grad():
    xs, pos_emb = pos_enc(x)
    print(f"Output xs: mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")

    # Check if xs == x * xscale (no positional encoding added)
    x_scaled = x * xscale
    diff_scaled_only = (xs - x_scaled).abs().max().item()
    print(f"\nMax diff (xs vs x*xscale): {diff_scaled_only:.6f}")

    # Check if xs == x * xscale + pos_emb[center:center+seq_len]
    # pos_emb is (1, 2*seq_len-1, d_model), center portion is pos_emb[:, seq_len-1:, :]
    seq_len = x.size(1)
    center = seq_len - 1  # For pos_emb of size 2*seq_len-1
    pos_slice = pos_emb[:, center:center+seq_len, :]
    x_with_pos = x * xscale + pos_slice
    diff_with_pos = (xs - x_with_pos).abs().max().item()
    print(f"Max diff (xs vs x*xscale + pos): {diff_with_pos:.6f}")

if diff_scaled_only < 0.001:
    print("\n=> Python does NOT add positional encoding, only scales x")
elif diff_with_pos < 0.001:
    print("\n=> Python DOES add positional encoding center slice")
else:
    print(f"\n=> Neither matches exactly")
