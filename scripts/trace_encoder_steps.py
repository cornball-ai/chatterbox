#!/usr/bin/env python3
"""Trace encoder forward step by step."""

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

outputs = {
    "input": test_input.cpu().float().contiguous(),
}

with torch.no_grad():
    # Make mask
    T = test_input.size(1)
    masks = torch.ones(1, 1, T, dtype=torch.bool, device='cuda')

    # 1. Embed
    result = encoder.embed(test_input, masks)
    xs = result[0]
    pos_emb = result[1]
    print(f"After embed: xs={xs.shape}, mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    outputs["after_embed"] = xs.cpu().float().contiguous()
    outputs["pos_emb"] = pos_emb.cpu().float().contiguous()

    # 2. Pre-lookahead
    xs = encoder.pre_lookahead_layer(xs)
    print(f"After pre_lookahead: mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    outputs["after_pre_lookahead"] = xs.cpu().float().contiguous()

    # 3. First encoder block - use the full block forward
    xs_enc0 = encoder.encoders[0](xs, pos_emb, masks[:, 0, :])
    print(f"After encoder[0]: mean={xs_enc0.mean().item():.6f}, std={xs_enc0.std().item():.6f}")
    outputs["after_encoder_0"] = xs_enc0.cpu().float().contiguous()

    # Run all encoder blocks
    xs_all = xs
    for i, enc in enumerate(encoder.encoders):
        xs_all = enc(xs_all, pos_emb, masks[:, 0, :])
        if i < 2:  # Only save first few
            outputs[f"after_encoder_{i}_full"] = xs_all.cpu().float().contiguous()

    print(f"\nAfter all encoders: mean={xs_all.mean().item():.6f}, std={xs_all.std().item():.6f}")
    outputs["after_all_encoders"] = xs_all.cpu().float().contiguous()

    # Upsample
    xs_up = encoder.up_layer(xs_all)
    print(f"After up_layer: {xs_up.shape}, mean={xs_up.mean().item():.6f}, std={xs_up.std().item():.6f}")
    outputs["after_up_layer"] = xs_up.cpu().float().contiguous()

    # Up embed
    masks_up = torch.ones(1, 1, xs_up.size(1), dtype=torch.bool, device='cuda')
    result_up = encoder.up_embed(xs_up, masks_up)
    xs_up = result_up[0]
    pos_emb_up = result_up[1]
    print(f"After up_embed: mean={xs_up.mean().item():.6f}, std={xs_up.std().item():.6f}")
    outputs["after_up_embed"] = xs_up.cpu().float().contiguous()

    # Up encoders
    for i, enc in enumerate(encoder.up_encoders):
        xs_up = enc(xs_up, pos_emb_up, masks_up[:, 0, :])
        if i < 2:
            outputs[f"after_up_encoder_{i}"] = xs_up.cpu().float().contiguous()

    print(f"After all up_encoders: mean={xs_up.mean().item():.6f}, std={xs_up.std().item():.6f}")
    outputs["after_all_up_encoders"] = xs_up.cpu().float().contiguous()

    # After norm
    xs_final = encoder.after_norm(xs_up)
    print(f"After after_norm: mean={xs_final.mean().item():.6f}, std={xs_final.std().item():.6f}")
    outputs["after_norm_final"] = xs_final.cpu().float().contiguous()

save_file(outputs, "/outputs/encoder_steps.safetensors")
print("\nSaved to /outputs/encoder_steps.safetensors")
