#!/usr/bin/env python3
"""Trace full estimator including mid blocks and up block."""

import torch
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
from einops import rearrange

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator
estimator.eval()

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
    t_emb = estimator.time_mlp(t_pos)

    # === Input packing ===
    time = x_in.shape[2]
    h = torch.cat([x_in, mu_in], dim=1)
    spks_exp = spks_in.unsqueeze(2).expand(-1, -1, time)
    h = torch.cat([h, spks_exp], dim=1)
    h = torch.cat([h, cond_in], dim=1)

    # === Down block ===
    down_block = estimator.down_blocks[0]
    down_resnet = down_block[0]
    h_res = down_resnet(h, mask_in, t_emb)

    h_attn = rearrange(h_res, "b c t -> b t c").contiguous()
    attn_mask = torch.zeros(2, 1, 50, 50, device='cuda')
    for transformer in down_block[1]:
        h_attn = transformer(hidden_states=h_attn, attention_mask=attn_mask, timestep=t_emb)
    h_attn = rearrange(h_attn, "b t c -> b c t").contiguous()

    hidden_skip = h_attn  # Save for skip connection
    h = down_block[2](h_attn * mask_in)  # down conv

    # === All mid blocks ===
    mask_mid = mask_in  # same mask
    for i, mid_block in enumerate(estimator.mid_blocks):
        mid_resnet = mid_block[0]
        h = mid_resnet(h, mask_mid, t_emb)

        h_mid_attn = rearrange(h, "b c t -> b t c").contiguous()
        for transformer in mid_block[1]:
            h_mid_attn = transformer(hidden_states=h_mid_attn, attention_mask=attn_mask, timestep=t_emb)
        h = rearrange(h_mid_attn, "b t c -> b c t").contiguous()

        print(f"mid[{i}]: mean={h.mean().item():.6f}")
        outputs_dict[f"mid{i}"] = h.cpu().contiguous()

    # === Up block ===
    up_block = estimator.up_blocks[0]
    up_conv = up_block[0]
    up_resnet = up_block[1]
    up_transformers = up_block[2]

    h = up_conv(h * mask_mid)
    print(f"after up_conv: mean={h.mean().item():.6f}")
    outputs_dict["up_conv"] = h.cpu().contiguous()

    # Skip connection
    h = torch.cat([h, hidden_skip], dim=1)  # Double channels
    print(f"after skip cat: mean={h.mean().item():.6f}, shape={h.shape}")
    outputs_dict["up_skip_cat"] = h.cpu().contiguous()

    h = up_resnet(h, mask_mid, t_emb)
    print(f"after up_resnet: mean={h.mean().item():.6f}")
    outputs_dict["up_resnet"] = h.cpu().contiguous()

    h_up_attn = rearrange(h, "b c t -> b t c").contiguous()
    for i, transformer in enumerate(up_transformers):
        h_up_attn = transformer(hidden_states=h_up_attn, attention_mask=attn_mask, timestep=t_emb)
        print(f"  up_transformer[{i}]: mean={h_up_attn.mean().item():.6f}")
    h = rearrange(h_up_attn, "b t c -> b c t").contiguous()
    print(f"after up_transformers: mean={h.mean().item():.6f}")
    outputs_dict["up_attn"] = h.cpu().contiguous()

    # === Final block ===
    h = estimator.final_block(h, mask_mid)
    print(f"after final_block: mean={h.mean().item():.6f}")
    outputs_dict["final_block"] = h.cpu().contiguous()

    h = estimator.final_proj(h * mask_mid)
    print(f"after final_proj: mean={h.mean().item():.6f}")
    outputs_dict["output"] = h.cpu().contiguous()

    save_file(outputs_dict, "/outputs/estimator_trace_full.safetensors")
    print(f"\nSaved to estimator_trace_full.safetensors")
