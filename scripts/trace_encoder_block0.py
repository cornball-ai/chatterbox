#!/usr/bin/env python3
"""Trace encoder block 0 in detail."""

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

print(f"Input: {test_input.shape}")

outputs = {}

with torch.no_grad():
    # Get masks
    T = test_input.size(1)
    masks = torch.ones(1, 1, T, dtype=torch.bool, device='cuda')

    # Run through embed and pre_lookahead
    result = encoder.embed(test_input, masks)
    xs = result[0]
    pos_emb = result[1]
    xs = encoder.pre_lookahead_layer(xs)

    print(f"Before encoder_0: xs mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    print(f"pos_emb shape: {pos_emb.shape}, mean={pos_emb.mean().item():.6f}")

    outputs["pre_lookahead"] = xs.cpu().float().contiguous()
    outputs["pos_emb"] = pos_emb.cpu().float().contiguous()

    # Get encoder block 0
    enc0 = encoder.encoders[0]

    # Step through manually
    residual = xs
    xs_normed = enc0.norm_mha(xs)
    print(f"\nAfter norm_mha: mean={xs_normed.mean().item():.6f}, std={xs_normed.std().item():.6f}")
    outputs["norm_mha_0"] = xs_normed.cpu().float().contiguous()

    # Self-attention forward
    # In Python: self.self_attn(x, x, x, mask, pos_emb)
    # Mask shape should be (1, 1, T) for the full encoder block
    # But for attention, it uses (1, 1, T, T) or handles it internally
    # Let's just run the full encoder block instead
    enc0_out = enc0(xs, pos_emb, masks[:, 0, :])
    attn_out = enc0_out  # The output is the final output, not attn_out
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    print(f"After encoder_0 (full): mean={attn_out.mean().item():.6f}, std={attn_out.std().item():.6f}")
    outputs["encoder_0_output"] = attn_out.cpu().float().contiguous()
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    print(f"After self_attn: mean={attn_out.mean().item():.6f}, std={attn_out.std().item():.6f}")
    outputs["self_attn_0"] = attn_out.cpu().float().contiguous()

    # Dropout is in eval mode, so no change
    xs = residual + attn_out
    print(f"After attn+residual: mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    outputs["after_attn_residual_0"] = xs.cpu().float().contiguous()

    # Feed-forward
    residual = xs
    xs_normed = enc0.norm_ff(xs)
    print(f"After norm_ff: mean={xs_normed.mean().item():.6f}, std={xs_normed.std().item():.6f}")
    outputs["norm_ff_0"] = xs_normed.cpu().float().contiguous()

    ff_out = enc0.feed_forward(xs_normed)
    print(f"After feed_forward: mean={ff_out.mean().item():.6f}, std={ff_out.std().item():.6f}")
    outputs["feed_forward_0"] = ff_out.cpu().float().contiguous()

    xs = residual + ff_out
    print(f"After ff+residual: mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    outputs["encoder_0_output"] = xs.cpu().float().contiguous()

save_file(outputs, "/outputs/encoder_block0.safetensors")
print("\nSaved to /outputs/encoder_block0.safetensors")
