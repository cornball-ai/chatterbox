#!/usr/bin/env python3
"""Save positional encoding intermediates for debugging."""

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

# Trace the embed layer
with torch.no_grad():
    # Get mask
    T = test_input.size(1)
    masks = torch.ones(1, 1, T, dtype=torch.bool, device='cuda')

    # Step through embed.out (linear + layernorm)
    x = test_input
    for name, module in encoder.embed.out._modules.items():
        x = module(x)
        print(f"After {name}: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

    # Now the positional encoding
    pos_enc = encoder.embed.pos_enc
    print(f"\npos_enc type: {type(pos_enc).__name__}")
    print(f"pos_enc.xscale: {pos_enc.xscale}")

    # Get the PE buffer
    pe = pos_enc.pe
    print(f"pe buffer shape: {pe.shape}")
    print(f"pe mean: {pe.mean().item():.6f}, std: {pe.std().item():.6f}")

    # Run pos_enc forward
    xs, pos_emb = pos_enc(x)
    print(f"\nAfter pos_enc:")
    print(f"  xs shape: {xs.shape}, mean={xs.mean().item():.6f}, std={xs.std().item():.6f}")
    print(f"  pos_emb shape: {pos_emb.shape}, mean={pos_emb.mean().item():.6f}, std={pos_emb.std().item():.6f}")

    # Check the slicing indices
    seq_len = x.size(1)
    max_len = pe.size(1)
    center = max_len // 2  # This is how Espnet does it
    start_idx = center - seq_len + 1
    end_idx = center + seq_len
    print(f"\nSlicing: seq_len={seq_len}, max_len={max_len}, center={center}")
    print(f"  indices: [{start_idx}:{end_idx}] (length {end_idx - start_idx})")

    # Get the slice manually to verify
    pos_emb_manual = pe[:, start_idx:end_idx, :]
    print(f"  manual slice shape: {pos_emb_manual.shape}")
    diff = (pos_emb.cpu() - pos_emb_manual.cpu()).abs().max().item()
    print(f"  diff from pos_enc output: {diff}")

    # Also save the pe buffer at key positions
    print(f"\nPE buffer samples:")
    print(f"  pe[0, 0, :5]: {pe[0, 0, :5].tolist()}")
    print(f"  pe[0, center, :5]: {pe[0, center, :5].tolist()}")
    print(f"  pe[0, -1, :5]: {pe[0, -1, :5].tolist()}")

# Save for comparison
outputs = {
    "x_after_embed_out": x.cpu().float().contiguous(),
    "xs_after_pos_enc": xs.cpu().float().contiguous(),
    "pos_emb": pos_emb.cpu().float().contiguous(),
    "pe_buffer": pe.cpu().float().contiguous(),
}

save_file(outputs, "/outputs/pos_emb_reference.safetensors")
print("\nSaved to /outputs/pos_emb_reference.safetensors")
