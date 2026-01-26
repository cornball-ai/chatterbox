#!/usr/bin/env python3
"""Trace encoder block 0 using hooks."""

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

# Register hooks on encoder block 0
enc0 = encoder.encoders[0]
hooks = []

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            outputs[name] = output[0].detach().cpu().float().contiguous()
        else:
            outputs[name] = output.detach().cpu().float().contiguous()
        print(f"{name}: mean={outputs[name].mean().item():.6f}, std={outputs[name].std().item():.6f}")
    return hook

hooks.append(enc0.norm_mha.register_forward_hook(make_hook("norm_mha")))
hooks.append(enc0.self_attn.register_forward_hook(make_hook("self_attn")))
hooks.append(enc0.norm_ff.register_forward_hook(make_hook("norm_ff")))
hooks.append(enc0.feed_forward.register_forward_hook(make_hook("feed_forward")))
hooks.append(enc0.register_forward_hook(make_hook("encoder_0")))

with torch.no_grad():
    # Run through to encoder_0
    T = test_input.size(1)
    masks = torch.ones(1, 1, T, dtype=torch.bool, device='cuda')

    result = encoder.embed(test_input, masks)
    xs = result[0]
    pos_emb = result[1]
    xs = encoder.pre_lookahead_layer(xs)

    outputs["pre_lookahead"] = xs.cpu().float().contiguous()
    outputs["pos_emb"] = pos_emb.cpu().float().contiguous()

    print(f"\nBefore encoder_0: mean={xs.mean().item():.6f}")
    print("\nRunning encoder_0...")

    # Run just encoder_0
    xs_out = enc0(xs, pos_emb, masks[:, 0, :])

# Clean up hooks
for h in hooks:
    h.remove()

save_file(outputs, "/outputs/encoder_block0.safetensors")
print("\nSaved to /outputs/encoder_block0.safetensors")
