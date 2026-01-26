#!/usr/bin/env python3
"""Save attention layer intermediates for debugging."""

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

# Run through embed and pre_lookahead
with torch.no_grad():
    # Get mask
    T = test_input.size(1)
    masks = torch.ones(1, 1, T, dtype=torch.bool, device='cuda')

    # Embed
    embed_result = encoder.embed(test_input, masks)
    xs = embed_result[0]
    pos_emb = embed_result[1]

    # Pre-lookahead
    xs = encoder.pre_lookahead_layer(xs)

    print(f"After pre_lookahead: xs={xs.shape}, pos_emb={pos_emb.shape}")

    # Get first encoder block's attention
    enc_block = encoder.encoders[0]
    attn = enc_block.self_attn

    # Apply norm
    xs_normed = enc_block.norm_mha(xs)
    print(f"After norm_mha: {xs_normed.shape}, mean={xs_normed.mean().item():.6f}")

    # Attention forward manually
    batch_size = xs_normed.size(0)
    seq_len = xs_normed.size(1)
    h = attn.h
    d_k = attn.d_k

    q = attn.linear_q(xs_normed).view(batch_size, -1, h, d_k).transpose(1, 2)
    k = attn.linear_k(xs_normed).view(batch_size, -1, h, d_k).transpose(1, 2)
    v = attn.linear_v(xs_normed).view(batch_size, -1, h, d_k).transpose(1, 2)

    print(f"q: {q.shape}, mean={q.mean().item():.6f}")
    print(f"k: {k.shape}, mean={k.mean().item():.6f}")

    p = attn.linear_pos(pos_emb).view(1, -1, h, d_k).transpose(1, 2)
    print(f"p: {p.shape}, mean={p.mean().item():.6f}")

    # Bias
    q_with_bias_u = q + attn.pos_bias_u.unsqueeze(0).unsqueeze(2)
    q_with_bias_v = q + attn.pos_bias_v.unsqueeze(0).unsqueeze(2)

    print(f"pos_bias_u: {attn.pos_bias_u.shape}, mean={attn.pos_bias_u.mean().item():.6f}")

    # Content attention
    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    print(f"matrix_ac: {matrix_ac.shape}, mean={matrix_ac.mean().item():.6f}")

    # Position attention
    matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
    print(f"matrix_bd before rel_shift: {matrix_bd.shape}, mean={matrix_bd.mean().item():.6f}")

    matrix_bd_shifted = attn.rel_shift(matrix_bd)
    print(f"matrix_bd after rel_shift: {matrix_bd_shifted.shape}, mean={matrix_bd_shifted.mean().item():.6f}")

    # Full scores
    scores = (matrix_ac + matrix_bd_shifted) / (d_k ** 0.5)
    print(f"scores: {scores.shape}, mean={scores.mean().item():.6f}")

# Save intermediates
outputs = {
    "xs_normed": xs_normed.cpu().float().contiguous(),
    "q": q.cpu().float().contiguous(),
    "k": k.cpu().float().contiguous(),
    "v": v.cpu().float().contiguous(),
    "p": p.cpu().float().contiguous(),
    "pos_bias_u": attn.pos_bias_u.cpu().float().contiguous(),
    "pos_bias_v": attn.pos_bias_v.cpu().float().contiguous(),
    "matrix_ac": matrix_ac.cpu().float().contiguous(),
    "matrix_bd_before": matrix_bd.cpu().float().contiguous(),
    "matrix_bd_after": matrix_bd_shifted.cpu().float().contiguous(),
    "scores": scores.cpu().float().contiguous(),
}

save_file(outputs, "/outputs/attn_reference.safetensors")
print("\nSaved to /outputs/attn_reference.safetensors")
