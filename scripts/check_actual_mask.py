#!/usr/bin/env python3
"""Check actual mask used during encoder forward."""

import torch
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained('cuda')
encoder = tts.s3gen.flow.encoder
encoder.eval()

# Check encoder config
print("Encoder config:")
print(f"  use_dynamic_chunk: {encoder.use_dynamic_chunk}")
print(f"  use_dynamic_left_chunk: {encoder.use_dynamic_left_chunk}")
print(f"  static_chunk_size: {encoder.static_chunk_size}")

# Create test input
torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (1, 50), device='cuda')
test_input = tts.s3gen.flow.input_embedding(test_tokens)
test_lens = torch.tensor([50], device='cuda')

# Hook to capture chunk_masks
captured = {}

def hook(module, args, kwargs):
    # forward_layers is called with (xs, chunk_masks, pos_emb, mask_pad)
    captured['chunk_masks'] = args[1].detach().cpu()
    captured['mask_pad'] = args[3].detach().cpu()
    return None

# Can't easily hook forward_layers since it's not a module
# Instead, hook the first encoder layer to see what it receives

def enc_hook(module, args):
    captured['enc0_x'] = args[0].detach().cpu()
    captured['enc0_mask'] = args[1].detach().cpu()
    captured['enc0_pos_emb'] = args[2].detach().cpu()
    if len(args) > 3:
        captured['enc0_mask_pad'] = args[3].detach().cpu()
    return None

handle = encoder.encoders[0].register_forward_pre_hook(enc_hook)

with torch.no_grad():
    output, output_lens = encoder(test_input, test_lens)

handle.remove()

print("\nEncoder layer 0 received:")
for name, tensor in captured.items():
    print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
    if tensor.numel() < 100:
        print(f"    values: {tensor}")
    else:
        print(f"    mean={tensor.float().mean().item():.6f}")
