#!/usr/bin/env python3
"""Trace estimator intermediate values step by step."""

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
    # === 1. Time embedding ===
    t_pos = estimator.time_embeddings(t_in)
    print(f"\n1. time_embeddings: shape={t_pos.shape}, mean={t_pos.mean().item():.6f}, std={t_pos.std().item():.6f}")
    outputs_dict["time_pos"] = t_pos.cpu().contiguous()

    t_emb = estimator.time_mlp(t_pos)
    print(f"2. time_mlp: shape={t_emb.shape}, mean={t_emb.mean().item():.6f}, std={t_emb.std().item():.6f}")
    outputs_dict["time_emb"] = t_emb.cpu().contiguous()

    # === 2. Input packing ===
    time = x_in.shape[2]  # 50
    h = torch.cat([x_in, mu_in], dim=1)  # [2, 160, 50]
    spks_exp = spks_in.unsqueeze(2).expand(-1, -1, time)  # [2, 80, 50]
    h = torch.cat([h, spks_exp], dim=1)  # [2, 240, 50]
    h = torch.cat([h, cond_in], dim=1)  # [2, 320, 50]
    print(f"3. Packed h: shape={h.shape}, mean={h.mean().item():.6f}")
    outputs_dict["packed_h"] = h.cpu().contiguous()

    # === 3. Down block ===
    down_block = estimator.down_blocks[0]
    down_resnet = down_block[0]

    # Resnet forward(x, mask, time_emb)
    h_res = down_resnet(h, mask_in, t_emb)
    print(f"4. down_resnet: shape={h_res.shape}, mean={h_res.mean().item():.6f}, std={h_res.std().item():.6f}")
    outputs_dict["down_resnet"] = h_res.cpu().contiguous()

    # Attention mask - use None/zeros for comparison
    from einops import rearrange

    h_attn = rearrange(h_res, "b c t -> b t c").contiguous()
    # Use zeros attention mask for cleaner comparison
    attn_mask = torch.zeros(2, 1, 50, 50, device='cuda')
    print(f"5. attn_mask: zeros shape={attn_mask.shape}")
    outputs_dict["attn_mask"] = attn_mask.cpu().contiguous()

    # Transformer blocks
    down_transformers = down_block[1]
    for i, transformer in enumerate(down_transformers):
        h_attn = transformer(hidden_states=h_attn, attention_mask=attn_mask, timestep=t_emb)
        print(f"   transformer[{i}]: mean={h_attn.mean().item():.6f}")

    h_attn = rearrange(h_attn, "b t c -> b c t").contiguous()
    print(f"6. after transformers: mean={h_attn.mean().item():.6f}, std={h_attn.std().item():.6f}")
    outputs_dict["down_attn"] = h_attn.cpu().contiguous()

    # Skip connection save
    hidden_skip = h_attn

    # Downsample
    down_conv = down_block[2]
    h_down = down_conv(h_attn * mask_in)
    print(f"7. down_conv: mean={h_down.mean().item():.6f}")
    outputs_dict["down_conv"] = h_down.cpu().contiguous()

    # === 4. Mid block 0 ===
    mid_block = estimator.mid_blocks[0]
    mid_resnet = mid_block[0]
    h_mid = mid_resnet(h_down, mask_in, t_emb)
    print(f"8. mid[0]_resnet: mean={h_mid.mean().item():.6f}")
    outputs_dict["mid0_resnet"] = h_mid.cpu().contiguous()

    # === 5. Full forward for comparison ===
    output = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
    print(f"\nFinal output: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
    outputs_dict["output"] = output.cpu().contiguous()

    save_file(outputs_dict, "/outputs/estimator_trace.safetensors")
    print(f"\nSaved to estimator_trace.safetensors")
