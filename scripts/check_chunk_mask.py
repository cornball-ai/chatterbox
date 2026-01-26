#!/usr/bin/env python3
"""Check what attention mask add_optional_chunk_mask creates."""

import torch
import sys

# Try to import from different locations
try:
    from cosyvoice.transformer.embedding import add_optional_chunk_mask, mask_to_bias
except ImportError:
    print("cosyvoice not available, checking chatterbox internals...")
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained("cuda")
    estimator = model.s3gen.flow.decoder.estimator

    # Hook to capture attention mask
    attn_masks = []
    def hook(module, input, output):
        if hasattr(module, '_last_attn_mask'):
            attn_masks.append(module._last_attn_mask.cpu())

    # We need to trace inside the forward to see the mask
    # Let's just run the model and print statistics
    torch.manual_seed(42)
    with torch.no_grad():
        batch = 2
        time = 50

        x = torch.randn(batch, 80, time, device='cuda')
        mask = torch.ones(batch, 1, time, device='cuda')
        mu = torch.randn(batch, 80, time, device='cuda')
        t = torch.ones(batch, device='cuda') * 0.5
        spks = torch.randn(batch, 80, device='cuda')
        cond = torch.randn(batch, 80, time, device='cuda')

        # Forward with static_chunk_size=0
        print(f"static_chunk_size: {estimator.static_chunk_size}")
        out1 = estimator(x, mask, mu, t, spks, cond)
        print(f"Output with static_chunk_size=0: mean={out1.mean().item():.6f}")

        # Try to change static_chunk_size and compare
        old_chunk_size = estimator.static_chunk_size
        estimator.static_chunk_size = -1
        out2 = estimator(x, mask, mu, t, spks, cond)
        print(f"Output with static_chunk_size=-1: mean={out2.mean().item():.6f}")

        diff = (out1 - out2).abs().max().item()
        print(f"Difference: {diff:.6f}")

        # Restore
        estimator.static_chunk_size = old_chunk_size
    sys.exit()

# If we have cosyvoice, test the function
print("Testing add_optional_chunk_mask...")
device = 'cpu'
batch = 1
time = 10

x = torch.randn(batch, time, 256, device=device)
mask = torch.ones(batch, 1, time, dtype=torch.bool, device=device)

for chunk_size in [-1, 0, 4]:
    attn_mask = add_optional_chunk_mask(x, mask, False, False, 0, chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
    print(f"\nchunk_size={chunk_size}:")
    print(f"  shape: {attn_mask.shape}")
    print(f"  unique values: {attn_mask.unique()}")
    if attn_mask.shape[-1] <= 10:
        print(f"  mask[0,0]: {attn_mask[0, 0]}")
