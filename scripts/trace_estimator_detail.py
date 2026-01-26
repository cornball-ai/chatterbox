#!/usr/bin/env python3
"""Trace estimator intermediate values for comparison."""

import torch
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator
estimator.eval()

# Load the Euler step trace inputs
euler_ref = load_file("/outputs/euler_step_trace.safetensors")

x_in = euler_ref["step0_x_in"].cuda()
mu_in = euler_ref["step0_mu_in"].cuda()
t_in = euler_ref["step0_t_in"].cuda()
spks_in = euler_ref["step0_spks_in"].cuda()
cond_in = euler_ref["step0_cond_in"].cuda()
mask_in = torch.ones(2, 1, 50, device='cuda')

outputs_dict = {}

with torch.no_grad():
    # === Time embedding ===
    t_pos = estimator.time_embeddings(t_in)
    print(f"time_embeddings output: shape={t_pos.shape}, mean={t_pos.mean().item():.6f}")
    outputs_dict["time_pos"] = t_pos.cpu().contiguous()

    t_emb = estimator.time_mlp(t_pos)
    print(f"time_mlp output: shape={t_emb.shape}, mean={t_emb.mean().item():.6f}")
    outputs_dict["time_embed"] = t_emb.cpu().contiguous()

    # === Input packing ===
    time = x_in.shape[2]  # 50
    h = torch.cat([x_in, mu_in], dim=1)  # [2, 160, 50]
    spks_exp = spks_in.unsqueeze(2).expand(-1, -1, time)  # [2, 80, 50]
    h = torch.cat([h, spks_exp], dim=1)  # [2, 240, 50]
    h = torch.cat([h, cond_in], dim=1)  # [2, 320, 50]
    print(f"Packed h: shape={h.shape}, mean={h.mean().item():.6f}")
    outputs_dict["packed_h"] = h.cpu().contiguous()

    # === Input projection (first conv) ===
    # estimator.input_blocks[0] should be a CausalBlock1D
    input_block = estimator.input_blocks[0]
    h_proj = input_block(h, mask_in)
    print(f"After input_blocks[0]: shape={h_proj.shape}, mean={h_proj.mean().item():.6f}")
    outputs_dict["after_input_block"] = h_proj.cpu().contiguous()

    # === Down block ===
    down_block = estimator.down_blocks[0]
    # ResnetBlock
    h_res = down_block.resnet(h_proj, t_emb, mask_in)
    print(f"After down_block resnet: shape={h_res.shape}, mean={h_res.mean().item():.6f}")
    outputs_dict["down_resnet"] = h_res.cpu().contiguous()

    # Attention blocks in down_block
    h_attn = h_res.transpose(1, 2)  # [B, T, C]
    for i, attn_block in enumerate(down_block.attentions):
        h_attn = attn_block(h_attn)
        print(f"After down_block attention[{i}]: mean={h_attn.mean().item():.6f}")
    h_attn = h_attn.transpose(1, 2)  # [B, C, T]
    outputs_dict["down_attn"] = h_attn.cpu().contiguous()

    # Downsample conv
    h_down = down_block.downsample(h_attn, mask_in)
    print(f"After down_block downsample: shape={h_down.shape}, mean={h_down.mean().item():.6f}")
    outputs_dict["down_output"] = h_down.cpu().contiguous()

    # === First mid block ===
    mid_block = estimator.mid_blocks[0]
    h_mid_res = mid_block.resnet(h_down, t_emb, mask_in)
    print(f"After mid_block[0] resnet: shape={h_mid_res.shape}, mean={h_mid_res.mean().item():.6f}")
    outputs_dict["mid0_resnet"] = h_mid_res.cpu().contiguous()

    h_mid_attn = h_mid_res.transpose(1, 2)
    for i, attn_block in enumerate(mid_block.attentions):
        h_mid_attn = attn_block(h_mid_attn)
    h_mid_attn = h_mid_attn.transpose(1, 2)
    print(f"After mid_block[0] attention: mean={h_mid_attn.mean().item():.6f}")
    outputs_dict["mid0_attn"] = h_mid_attn.cpu().contiguous()

    # Save for R comparison
    save_file(outputs_dict, "/outputs/estimator_trace.safetensors")
    print(f"\nSaved to estimator_trace.safetensors")
