#!/usr/bin/env python3
"""Trace rel_shift in detail."""

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

with torch.no_grad():
    T = test_input.size(1)
    masks = torch.ones(1, 1, T, dtype=torch.bool, device='cuda')

    # Get through to attention input
    result = encoder.embed(test_input, masks)
    xs = result[0]
    pos_emb = result[1]
    xs = encoder.pre_lookahead_layer(xs)

    # Run first encoder block manually
    enc0 = encoder.encoders[0]
    xs_normed = enc0.norm_mha(xs)
    attn = enc0.self_attn

    # Get q, k, v, p
    batch_size = xs_normed.size(0)
    seq_len = xs_normed.size(1)

    q = attn.linear_q(xs_normed).view(batch_size, -1, attn.h, attn.d_k).transpose(1, 2)
    k = attn.linear_k(xs_normed).view(batch_size, -1, attn.h, attn.d_k).transpose(1, 2)
    v = attn.linear_v(xs_normed).view(batch_size, -1, attn.h, attn.d_k).transpose(1, 2)
    p = attn.linear_pos(pos_emb).view(1, -1, attn.h, attn.d_k).transpose(1, 2)

    print(f"q: {q.shape}, mean={q.mean().item():.6f}")
    print(f"p: {p.shape}, mean={p.mean().item():.6f}")

    # Position bias
    q_with_bias_u = q + attn.pos_bias_u.unsqueeze(0).unsqueeze(2)
    q_with_bias_v = q + attn.pos_bias_v.unsqueeze(0).unsqueeze(2)

    # Content attention
    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    print(f"matrix_ac: {matrix_ac.shape}, mean={matrix_ac.mean().item():.6f}")

    # Position attention before shift
    matrix_bd_before = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
    print(f"matrix_bd_before: {matrix_bd_before.shape}, mean={matrix_bd_before.mean().item():.6f}")

    # Apply rel_shift
    matrix_bd_after = attn.rel_shift(matrix_bd_before)
    print(f"matrix_bd_after: {matrix_bd_after.shape}, mean={matrix_bd_after.mean().item():.6f}")

    # Check rel_shift implementation
    import inspect
    print("\nrel_shift source:")
    src = inspect.getsource(attn.rel_shift)
    for line in src.split('\n')[:20]:
        print(f"  {line}")

# Save
outputs = {
    "q": q.cpu().float().contiguous(),
    "p": p.cpu().float().contiguous(),
    "q_with_bias_v": q_with_bias_v.cpu().float().contiguous(),
    "matrix_ac": matrix_ac.cpu().float().contiguous(),
    "matrix_bd_before": matrix_bd_before.cpu().float().contiguous(),
    "matrix_bd_after": matrix_bd_after.cpu().float().contiguous(),
}

save_file(outputs, "/outputs/rel_shift_reference.safetensors")
print("\nSaved to /outputs/rel_shift_reference.safetensors")
