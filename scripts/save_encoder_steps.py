#!/usr/bin/env python3
"""Save encoder step-by-step outputs for R validation."""

import torch
import torch.nn.functional as F
import safetensors.torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3gen.transformer.upsample_encoder import add_optional_chunk_mask

# Helper function
def make_pad_mask(lengths, max_len=None):
    """Create boolean mask for padding positions."""
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    lengths_expand = lengths.unsqueeze(1).expand(batch_size, max_len)
    return seq_range_expand >= lengths_expand

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
encoder = model.s3gen.flow.encoder
encoder.eval()

# Create test input (token embeddings from input_embedding)
batch_size = 1
seq_len = 32
input_dim = 512

torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, input_dim).cuda()
x_lens = torch.tensor([seq_len]).cuda()

print(f"Input x shape: {x.shape}")
print(f"Input x_lens: {x_lens}")

outputs = {}
outputs['input_x'] = x.cpu()
outputs['input_x_lens'] = x_lens.cpu()

with torch.no_grad():
    # Following encoder.forward exactly
    T = x.size(1)
    masks = ~make_pad_mask(x_lens, T).unsqueeze(1)  # (B, 1, T)
    print(f"masks shape: {masks.shape}")
    outputs['masks'] = masks.cpu().float()

    # Step 1: embed
    print("\n=== Step 1: embed ===")
    xs, pos_emb, masks_emb = encoder.embed(x, masks)
    print(f"xs shape after embed: {xs.shape}")
    print(f"pos_emb shape: {pos_emb.shape}")
    outputs['after_embed_xs'] = xs.cpu()
    outputs['after_embed_pos_emb'] = pos_emb.cpu()

    # Step 2: Create chunk masks
    mask_pad = masks_emb
    chunk_masks = add_optional_chunk_mask(
        xs, masks_emb,
        encoder.use_dynamic_chunk,
        encoder.use_dynamic_left_chunk,
        0,  # decoding_chunk_size
        encoder.static_chunk_size,
        -1  # num_decoding_left_chunks
    )
    print(f"chunk_masks shape: {chunk_masks.shape}")
    outputs['chunk_masks'] = chunk_masks.cpu().float()

    # Step 3: pre_lookahead_layer
    print("\n=== Step 3: pre_lookahead_layer ===")
    xs_pll = encoder.pre_lookahead_layer(xs)
    print(f"xs after pre_lookahead: {xs_pll.shape}")
    outputs['after_pll_xs'] = xs_pll.cpu()

    # Step 4: forward_layers (6 Conformer blocks)
    print("\n=== Step 4: forward_layers (6 Conformer blocks) ===")
    xs_enc = xs_pll
    for i, layer in enumerate(encoder.encoders):
        result = layer(xs_enc, chunk_masks, pos_emb, mask_pad)
        xs_enc = result[0]
        print(f"xs after block {i}: mean={xs_enc.mean().item():.4f}, std={xs_enc.std().item():.4f}")
    outputs['after_encoders_xs'] = xs_enc.cpu()

    # Step 5: Transpose for upsample
    print("\n=== Step 5: Transpose + Upsample ===")
    xs_t = xs_enc.transpose(1, 2).contiguous()  # [B, T, D] -> [B, D, T]
    print(f"xs transposed: {xs_t.shape}")
    outputs['before_upsample_xs'] = xs_t.cpu()

    # Step 6: Upsample
    xs_up, xs_up_lens = encoder.up_layer(xs_t, x_lens)
    print(f"xs after upsample: {xs_up.shape}")
    print(f"xs_up_lens: {xs_up_lens}")
    outputs['after_upsample_xs'] = xs_up.cpu()
    outputs['after_upsample_lens'] = xs_up_lens.cpu()

    # Step 7: Transpose back
    xs_up_t = xs_up.transpose(1, 2).contiguous()  # [B, D, T] -> [B, T, D]
    print(f"xs transposed back: {xs_up_t.shape}")
    outputs['after_upsample_t_xs'] = xs_up_t.cpu()

    # Step 8: up_embed
    print("\n=== Step 8: up_embed ===")
    T_up = xs_up_t.size(1)
    masks_up = ~make_pad_mask(xs_up_lens, T_up).unsqueeze(1)
    xs_up2, pos_emb_up, masks_up2 = encoder.up_embed(xs_up_t, masks_up)
    print(f"xs after up_embed: {xs_up2.shape}")
    print(f"pos_emb_up shape: {pos_emb_up.shape}")
    outputs['after_up_embed_xs'] = xs_up2.cpu()
    outputs['after_up_embed_pos_emb'] = pos_emb_up.cpu()

    # Step 9: Create chunk masks for up_encoders
    mask_pad_up = masks_up2
    chunk_masks_up = add_optional_chunk_mask(
        xs_up2, masks_up2,
        encoder.use_dynamic_chunk,
        encoder.use_dynamic_left_chunk,
        0,
        encoder.static_chunk_size * encoder.up_layer.stride,
        -1
    )

    # Step 10: forward_up_layers (4 Conformer blocks)
    print("\n=== Step 10: forward_up_layers (4 Conformer blocks) ===")
    xs_up_enc = xs_up2
    for i, layer in enumerate(encoder.up_encoders):
        result = layer(xs_up_enc, chunk_masks_up, pos_emb_up, mask_pad_up)
        xs_up_enc = result[0]
        print(f"xs after up_block {i}: mean={xs_up_enc.mean().item():.4f}, std={xs_up_enc.std().item():.4f}")
    outputs['after_up_encoders_xs'] = xs_up_enc.cpu()

    # Step 11: after_norm
    print("\n=== Step 11: after_norm ===")
    xs_final = encoder.after_norm(xs_up_enc)
    print(f"xs after after_norm: {xs_final.shape}")
    print(f"Final: mean={xs_final.mean().item():.4f}, std={xs_final.std().item():.4f}")
    outputs['after_norm_xs'] = xs_final.cpu()

    # Step 12: Full encoder forward
    print("\n=== Step 12: Full encoder forward ===")
    xs_full, masks_full = encoder(x, x_lens)
    print(f"Full encoder output: {xs_full.shape}")
    outputs['full_encoder_xs'] = xs_full.cpu()

    # Verify step-by-step matches full forward
    diff = (xs_final - xs_full).abs().max().item()
    print(f"\nStep-by-step vs full forward max diff: {diff}")

# Save outputs
output_path = "/workspace/outputs/encoder_steps.safetensors"
safetensors.torch.save_file(outputs, output_path)
print(f"\nSaved {len(outputs)} tensors to {output_path}")
for k, v in outputs.items():
    print(f"  {k}: {list(v.shape)}")
