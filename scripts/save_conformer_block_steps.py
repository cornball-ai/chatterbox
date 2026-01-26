#!/usr/bin/env python3
"""Save conformer block 0 intermediate outputs for comparison."""

import torch
import safetensors.torch
from chatterbox.tts import ChatterboxTTS

# Load reference inputs
ref = safetensors.torch.load_file("/workspace/outputs/encoder_steps.safetensors")

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
encoder = model.s3gen.flow.encoder
encoder.eval()

outputs = {}

with torch.no_grad():
    # Get inputs
    x = ref['after_pll_xs'].cuda()
    pos_emb = ref['after_embed_pos_emb'].cuda()
    masks = ref['masks'].cuda().bool()

    print(f"Input x: {x.shape}, mean={x.mean().item():.6f}")
    print(f"Input pos_emb: {pos_emb.shape}")
    print(f"Input masks: {masks.shape}")

    outputs['input_x'] = x.cpu()
    outputs['input_pos_emb'] = pos_emb.cpu()
    outputs['input_masks'] = masks.cpu().float()

    # Get block 0
    block = encoder.encoders[0]

    # Step 1: Pre-norm for MHA
    print("\n=== Step 1: norm_mha ===")
    residual = x
    x_normed = block.norm_mha(x)
    print(f"After norm_mha: mean={x_normed.mean().item():.6f}, std={x_normed.std().item():.6f}")
    outputs['after_norm_mha'] = x_normed.cpu()

    # Step 2: Self-attention
    print("\n=== Step 2: self_attn ===")
    # The conformer self_attn forward signature:
    # forward(query, key, value, mask, pos_emb, cache=...)
    attn_out, _ = block.self_attn(x_normed, x_normed, x_normed, masks, pos_emb)
    print(f"After self_attn: mean={attn_out.mean().item():.6f}, std={attn_out.std().item():.6f}")
    outputs['after_self_attn'] = attn_out.cpu()

    # Step 3: Dropout (identity in eval)
    attn_out = block.dropout(attn_out)
    print(f"After dropout: mean={attn_out.mean().item():.6f}")

    # Step 4: Residual
    x_after_mha = residual + attn_out
    print(f"After MHA residual: mean={x_after_mha.mean().item():.6f}, std={x_after_mha.std().item():.6f}")
    outputs['after_mha_residual'] = x_after_mha.cpu()

    # Step 5: Pre-norm for FFN
    print("\n=== Step 5: norm_ff ===")
    residual = x_after_mha
    x_normed = block.norm_ff(x_after_mha)
    print(f"After norm_ff: mean={x_normed.mean().item():.6f}, std={x_normed.std().item():.6f}")
    outputs['after_norm_ff'] = x_normed.cpu()

    # Step 6: Feed-forward
    print("\n=== Step 6: feed_forward ===")
    ff_out = block.feed_forward(x_normed)
    print(f"After feed_forward: mean={ff_out.mean().item():.6f}, std={ff_out.std().item():.6f}")
    outputs['after_ff'] = ff_out.cpu()

    # Step 7: Dropout
    ff_out = block.dropout(ff_out)
    print(f"After dropout: mean={ff_out.mean().item():.6f}")

    # Step 8: Residual
    x_after_ff = residual + ff_out
    print(f"After FFN residual: mean={x_after_ff.mean().item():.6f}, std={x_after_ff.std().item():.6f}")
    outputs['after_ff_residual'] = x_after_ff.cpu()

    # Step 9: Full block output via forward
    print("\n=== Full block forward ===")
    x_full, _, _, _ = block(ref['after_pll_xs'].cuda(), masks, pos_emb, masks)
    print(f"Full block output: mean={x_full.mean().item():.6f}, std={x_full.std().item():.6f}")
    outputs['block0_full_output'] = x_full.cpu()

    # Check if step-by-step matches full
    diff = (x_after_ff - x_full).abs().max().item()
    print(f"\nStep-by-step vs full block diff: {diff}")

# Save
output_path = "/workspace/outputs/conformer_block0_steps.safetensors"
safetensors.torch.save_file(outputs, output_path)
print(f"\nSaved {len(outputs)} tensors to {output_path}")
