#!/usr/bin/env python3
"""Save attention intermediate outputs for comparison."""

import torch
import safetensors.torch
from chatterbox.tts import ChatterboxTTS

# Load reference inputs
ref = safetensors.torch.load_file("/workspace/outputs/conformer_block0_steps.safetensors")

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
encoder = model.s3gen.flow.encoder
encoder.eval()

outputs = {}

with torch.no_grad():
    x = ref['after_norm_mha'].cuda()  # Input to attention
    pos_emb = ref['input_pos_emb'].cuda()
    masks = ref['input_masks'].cuda().bool()

    print(f"Input x: {x.shape}, mean={x.mean().item():.6f}")
    print(f"Input pos_emb: {pos_emb.shape}")
    print(f"Input masks: {masks.shape}")

    outputs['attn_input_x'] = x.cpu()
    outputs['attn_input_pos_emb'] = pos_emb.cpu()
    outputs['attn_input_masks'] = masks.cpu().float()

    # Get attention module
    attn = encoder.encoders[0].self_attn

    # Step through attention forward
    query = x
    key = x
    value = x
    mask = masks
    cache = None

    batch_size = query.size(0)
    seq_len = query.size(1)
    n_head = attn.h
    d_k = attn.d_k

    print(f"\nbatch={batch_size}, seq={seq_len}, n_head={n_head}, d_k={d_k}")

    # Linear projections
    q = attn.linear_q(query).view(batch_size, -1, n_head, d_k).transpose(1, 2)
    k = attn.linear_k(key).view(batch_size, -1, n_head, d_k).transpose(1, 2)
    v = attn.linear_v(value).view(batch_size, -1, n_head, d_k).transpose(1, 2)

    print(f"\nq: {q.shape}, mean={q.mean().item():.6f}")
    print(f"k: {k.shape}, mean={k.mean().item():.6f}")
    print(f"v: {v.shape}, mean={v.mean().item():.6f}")

    outputs['attn_q'] = q.contiguous().cpu()
    outputs['attn_k'] = k.contiguous().cpu()
    outputs['attn_v'] = v.contiguous().cpu()

    # Position embedding projection
    p = attn.linear_pos(pos_emb).view(1, -1, n_head, d_k).transpose(1, 2)
    print(f"p: {p.shape}, mean={p.mean().item():.6f}")
    outputs['attn_p'] = p.contiguous().cpu()

    # Positional biases
    print(f"\npos_bias_u: {attn.pos_bias_u.shape}")
    print(f"pos_bias_v: {attn.pos_bias_v.shape}")
    outputs['attn_pos_bias_u'] = attn.pos_bias_u.cpu()
    outputs['attn_pos_bias_v'] = attn.pos_bias_v.cpu()

    # Query with positional bias
    # Python code: q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
    q_with_bias_u = q + attn.pos_bias_u.unsqueeze(0).unsqueeze(2)
    q_with_bias_v = q + attn.pos_bias_v.unsqueeze(0).unsqueeze(2)
    print(f"\nq_with_bias_u: {q_with_bias_u.shape}")
    outputs['attn_q_with_bias_u'] = q_with_bias_u.contiguous().cpu()
    outputs['attn_q_with_bias_v'] = q_with_bias_v.contiguous().cpu()

    # Content-based attention score
    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    print(f"matrix_ac: {matrix_ac.shape}, mean={matrix_ac.mean().item():.6f}")
    outputs['attn_matrix_ac'] = matrix_ac.cpu()

    # Position-based attention score
    matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
    print(f"matrix_bd (before rel_shift): {matrix_bd.shape}, mean={matrix_bd.mean().item():.6f}")
    outputs['attn_matrix_bd_before'] = matrix_bd.cpu()

    # Relative shift
    matrix_bd = attn.rel_shift(matrix_bd)
    print(f"matrix_bd (after rel_shift): {matrix_bd.shape}, mean={matrix_bd.mean().item():.6f}")
    outputs['attn_matrix_bd_after'] = matrix_bd.cpu()

    # Combined scores
    scores = (matrix_ac + matrix_bd) / (d_k ** 0.5)
    print(f"\nscores: {scores.shape}, mean={scores.mean().item():.6f}")
    outputs['attn_scores'] = scores.cpu()

    # Apply mask
    # Python: mask is (B, 1, T), expand to (B, 1, 1, T) for broadcasting
    mask_expanded = mask.unsqueeze(1)  # (B, 1, 1, T)
    print(f"mask_expanded: {mask_expanded.shape}")

    scores_masked = scores.masked_fill(~mask_expanded, float('-inf'))
    print(f"scores_masked mean: {scores_masked[0,0,0,:5]}")  # Show first few

    # Softmax
    attn_weights = torch.softmax(scores_masked, dim=-1)
    print(f"attn_weights: {attn_weights.shape}, mean={attn_weights.mean().item():.6f}")
    outputs['attn_weights'] = attn_weights.cpu()

    # Dropout (identity in eval)
    attn_weights = attn.dropout(attn_weights)

    # Apply to values
    output = torch.matmul(attn_weights, v)
    print(f"output before reshape: {output.shape}, mean={output.mean().item():.6f}")

    # Reshape
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, n_head * d_k)
    print(f"output after reshape: {output.shape}, mean={output.mean().item():.6f}")
    outputs['attn_output_before_proj'] = output.cpu()

    # Final projection
    output = attn.linear_out(output)
    print(f"final output: {output.shape}, mean={output.mean().item():.6f}")
    outputs['attn_final_output'] = output.cpu()

    # Compare with full forward
    full_output, _ = attn(x, x, x, masks, pos_emb)
    diff = (output - full_output).abs().max().item()
    print(f"\nStep-by-step vs full forward diff: {diff}")

# Save
output_path = "/workspace/outputs/attn_steps.safetensors"
safetensors.torch.save_file(outputs, output_path)
print(f"\nSaved {len(outputs)} tensors to {output_path}")
