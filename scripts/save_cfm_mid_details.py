#!/usr/bin/env python3
"""Save detailed mid block intermediates from Python."""

import torch
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
from einops import pack, repeat, rearrange

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator
estimator.eval()

# Use same random seed as before
torch.manual_seed(42)

with torch.no_grad():
    batch = 1
    time = 50

    # Create test inputs (same as cfm_estimator_steps.py)
    x = torch.randn(batch, 80, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    mu = torch.randn(batch, 80, time, device='cuda')
    t = torch.ones(batch, device='cuda') * 0.5
    spks = torch.randn(batch, 80, device='cuda')
    cond = torch.randn(batch, 80, time, device='cuda')

    outputs = {}

    # Time embedding
    t_emb = estimator.time_embeddings(t).to(t.dtype)
    t_emb = estimator.time_mlp(t_emb)
    outputs["time_emb"] = t_emb.cpu().contiguous()

    # Pack inputs
    h = pack([x, mu], "b * t")[0]
    spks_exp = repeat(spks, "b c -> b c t", t=time)
    h = pack([h, spks_exp], "b * t")[0]
    h = pack([h, cond], "b * t")[0]
    outputs["packed_all"] = h.cpu().contiguous()

    # Down block
    resnet, tfms, downsample = estimator.down_blocks[0]
    h = resnet(h, mask, t_emb)
    outputs["down_resnet"] = h.cpu().contiguous()

    h = rearrange(h, "b c t -> b t c").contiguous()

    # For full sequence (no streaming), attention mask is all zeros
    # This is what add_optional_chunk_mask returns for static_chunk_size=-1
    attn_mask = torch.zeros(batch, 1, time, time, device='cuda', dtype=h.dtype)
    outputs["attn_mask"] = attn_mask.cpu().contiguous()
    print(f"attn_mask: shape={attn_mask.shape}, all zeros")

    for i, tfm in enumerate(tfms):
        h = tfm(hidden_states=h, attention_mask=attn_mask, timestep=t_emb)
        if i == 0:
            outputs["down_tfm_0"] = h.cpu().contiguous()

    h = rearrange(h, "b t c -> b c t").contiguous()
    hidden_skip = h.clone()
    h = downsample(h * mask)
    outputs["downsample"] = h.cpu().contiguous()

    # Mid blocks - save all
    mask_mid = mask  # In Python the mask might change
    for i in range(12):
        resnet, tfms = estimator.mid_blocks[i]
        h = resnet(h, mask_mid, t_emb)
        outputs[f"mid_{i}_resnet"] = h.cpu().contiguous()

        h = rearrange(h, "b c t -> b t c").contiguous()
        # For full sequence, attention mask is zeros
        attn_mask = torch.zeros(batch, 1, h.size(1), h.size(1), device='cuda', dtype=h.dtype)

        for j, tfm in enumerate(tfms):
            h = tfm(hidden_states=h, attention_mask=attn_mask, timestep=t_emb)
            if j == 0:
                outputs[f"mid_{i}_tfm_0"] = h.cpu().contiguous()

        h = rearrange(h, "b t c -> b c t").contiguous()
        outputs[f"mid_{i}_after_tfms"] = h.cpu().contiguous()

    outputs["mid_final"] = h.cpu().contiguous()
    print(f"After all mid blocks: shape={h.shape}, mean={h.mean().item():.6f}, std={h.std().item():.6f}")

    # Up block
    resnet, tfms, upsample = estimator.up_blocks[0]
    h = upsample(h * mask)
    outputs["upsample"] = h.cpu().contiguous()

    h = torch.cat([h, hidden_skip], dim=1)
    outputs["skip_concat"] = h.cpu().contiguous()

    h = resnet(h, mask, t_emb)
    outputs["up_resnet"] = h.cpu().contiguous()

    h = rearrange(h, "b c t -> b t c").contiguous()
    attn_mask = torch.zeros(batch, 1, h.size(1), h.size(1), device='cuda', dtype=h.dtype)

    for j, tfm in enumerate(tfms):
        h = tfm(hidden_states=h, attention_mask=attn_mask, timestep=t_emb)

    h = rearrange(h, "b t c -> b c t").contiguous()
    outputs["up_after_tfms"] = h.cpu().contiguous()

    # Final
    h = estimator.final_block(h, mask)
    outputs["final_block"] = h.cpu().contiguous()

    h = estimator.final_proj(h * mask)
    outputs["final_proj"] = h.cpu().contiguous()

    print(f"Final output: mean={h.mean().item():.6f}, std={h.std().item():.6f}")

save_file(outputs, "/outputs/cfm_mid_details.safetensors")
print(f"Saved {len(outputs)} tensors to cfm_mid_details.safetensors")
