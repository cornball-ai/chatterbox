#!/usr/bin/env python3
"""Trace estimator intermediate values for comparison - v2."""

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

print(f"Inputs:")
print(f"  x_in: {x_in.shape}, mean={x_in.mean().item():.6f}")
print(f"  t_in: {t_in.shape}, values={t_in.cpu().numpy()}")

with torch.no_grad():
    # === Time embedding ===
    t_pos = estimator.time_embeddings(t_in)
    print(f"\n1. time_embeddings output: shape={t_pos.shape}, mean={t_pos.mean().item():.6f}, std={t_pos.std().item():.6f}")
    outputs_dict["time_pos"] = t_pos.cpu().contiguous()

    t_emb = estimator.time_mlp(t_pos)
    print(f"2. time_mlp output: shape={t_emb.shape}, mean={t_emb.mean().item():.6f}, std={t_emb.std().item():.6f}")
    outputs_dict["time_emb"] = t_emb.cpu().contiguous()

    # === Input packing ===
    time = x_in.shape[2]  # 50
    h = torch.cat([x_in, mu_in], dim=1)  # [2, 160, 50]
    spks_exp = spks_in.unsqueeze(2).expand(-1, -1, time)  # [2, 80, 50]
    h = torch.cat([h, spks_exp], dim=1)  # [2, 240, 50]
    h = torch.cat([h, cond_in], dim=1)  # [2, 320, 50]
    print(f"3. Packed h: shape={h.shape}, mean={h.mean().item():.6f}")
    outputs_dict["packed_h"] = h.cpu().contiguous()

    # === Down block ===
    down_block = estimator.down_blocks[0]

    # Get the resnet (index 0)
    down_resnet = down_block[0]
    # Resnet forward(x, temb, mask)
    h_res = down_resnet(h, t_emb, mask_in)
    print(f"4. After down resnet: shape={h_res.shape}, mean={h_res.mean().item():.6f}, std={h_res.std().item():.6f}")
    outputs_dict["down_resnet"] = h_res.cpu().contiguous()

    # Attention blocks (index 1 is ModuleList)
    down_attentions = down_block[1]
    h_attn = h_res.transpose(1, 2)  # [B, T, C]
    for i, attn_block in enumerate(down_attentions):
        h_attn_before = h_attn.clone()
        h_attn = attn_block(h_attn)
        print(f"   Attention[{i}]: mean_before={h_attn_before.mean().item():.6f}, mean_after={h_attn.mean().item():.6f}")
    h_attn = h_attn.transpose(1, 2)  # [B, C, T]
    print(f"5. After down attentions: mean={h_attn.mean().item():.6f}, std={h_attn.std().item():.6f}")
    outputs_dict["down_attn"] = h_attn.cpu().contiguous()

    # Store skip
    hidden_skip = h_attn

    # Downsample conv (index 2)
    down_conv = down_block[2]
    h_down = down_conv(h_attn * mask_in)
    print(f"6. After down conv: shape={h_down.shape}, mean={h_down.mean().item():.6f}")
    outputs_dict["down_output"] = h_down.cpu().contiguous()

    # === First mid block ===
    mid_block = estimator.mid_blocks[0]
    mid_resnet = mid_block[0]
    h_mid = mid_resnet(h_down, t_emb, mask_in)
    print(f"7. After mid[0] resnet: mean={h_mid.mean().item():.6f}, std={h_mid.std().item():.6f}")
    outputs_dict["mid0_resnet"] = h_mid.cpu().contiguous()

    mid_attentions = mid_block[1]
    h_mid_attn = h_mid.transpose(1, 2)
    for i, attn_block in enumerate(mid_attentions):
        h_mid_attn = attn_block(h_mid_attn)
    h_mid_attn = h_mid_attn.transpose(1, 2)
    print(f"8. After mid[0] attention: mean={h_mid_attn.mean().item():.6f}")
    outputs_dict["mid0_attn"] = h_mid_attn.cpu().contiguous()

    # Full forward for final output
    output = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
    print(f"\nFinal output: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
    outputs_dict["output"] = output.cpu().contiguous()

    # Save for R comparison
    save_file(outputs_dict, "/outputs/estimator_trace.safetensors")
    print(f"\nSaved to estimator_trace.safetensors")
