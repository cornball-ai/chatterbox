#!/usr/bin/env python3
"""Save CFM estimator intermediate values for R validation."""

import torch
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS
from einops import pack, repeat

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen
flow = s3gen.flow
decoder = flow.decoder
estimator = decoder.estimator

# Use consistent random seed
torch.manual_seed(42)

with torch.no_grad():
    batch = 1
    time = 50  # Smaller for easier debugging

    # Create test inputs
    x = torch.randn(batch, 80, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    mu = torch.randn(batch, 80, time, device='cuda')
    t = torch.ones(batch, device='cuda') * 0.5
    spks = torch.randn(batch, 80, device='cuda')
    cond = torch.randn(batch, 80, time, device='cuda')

    outputs = {
        "input_x": x.cpu().contiguous(),
        "input_mask": mask.cpu().contiguous(),
        "input_mu": mu.cpu().contiguous(),
        "input_t": t.cpu().contiguous(),
        "input_spks": spks.cpu().contiguous(),
        "input_cond": cond.cpu().contiguous(),
    }

    print(f"input_x: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
    print(f"input_mu: mean={mu.mean().item():.6f}, std={mu.std().item():.6f}")
    print(f"input_spks: mean={spks.mean().item():.6f}, std={spks.std().item():.6f}")
    print(f"input_cond: mean={cond.mean().item():.6f}, std={cond.std().item():.6f}")

    # Step 1: Time embedding
    t_emb = estimator.time_embeddings(t).to(t.dtype)
    outputs["time_emb_sinusoidal"] = t_emb.cpu().contiguous()
    print(f"time_emb_sinusoidal: {t_emb.shape}, mean={t_emb.mean().item():.6f}")

    t_emb = estimator.time_mlp(t_emb)
    outputs["time_emb_mlp"] = t_emb.cpu().contiguous()
    print(f"time_emb_mlp: {t_emb.shape}, mean={t_emb.mean().item():.6f}")

    # Step 2: Pack inputs
    x_packed = pack([x, mu], "b * t")[0]
    outputs["packed_x_mu"] = x_packed.cpu().contiguous()
    print(f"packed_x_mu: {x_packed.shape}, mean={x_packed.mean().item():.6f}")

    spks_exp = repeat(spks, "b c -> b c t", t=time)
    x_packed = pack([x_packed, spks_exp], "b * t")[0]
    outputs["packed_spks"] = x_packed.cpu().contiguous()
    print(f"packed_spks: {x_packed.shape}, mean={x_packed.mean().item():.6f}")

    x_packed = pack([x_packed, cond], "b * t")[0]
    outputs["packed_all"] = x_packed.cpu().contiguous()
    print(f"packed_all: {x_packed.shape}, mean={x_packed.mean().item():.6f}")

    # Run full forward and capture output
    full_output = estimator(x, mask, mu, t, spks, cond)
    outputs["full_output"] = full_output.cpu().contiguous()
    print(f"full_output: {full_output.shape}, mean={full_output.mean().item():.6f}, std={full_output.std().item():.6f}")

# Now use hooks to capture intermediate values
print("\n=== Using hooks to capture intermediates ===")
intermediates = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if hasattr(out, 'shape'):
            intermediates[name] = out.detach().cpu().contiguous()
            print(f"{name}: {out.shape}, mean={out.mean().item():.6f}")
    return hook

# Register hooks
hooks = []
hooks.append(estimator.down_blocks[0][0].register_forward_hook(make_hook("down_resnet")))
hooks.append(estimator.down_blocks[0][1][0].register_forward_hook(make_hook("down_tfm_0")))
hooks.append(estimator.down_blocks[0][2].register_forward_hook(make_hook("downsample")))

for i in range(12):
    hooks.append(estimator.mid_blocks[i][0].register_forward_hook(make_hook(f"mid_{i}_resnet")))
    hooks.append(estimator.mid_blocks[i][1][0].register_forward_hook(make_hook(f"mid_{i}_tfm_0")))

hooks.append(estimator.up_blocks[0][2].register_forward_hook(make_hook("upsample")))
hooks.append(estimator.up_blocks[0][0].register_forward_hook(make_hook("up_resnet")))
hooks.append(estimator.up_blocks[0][1][0].register_forward_hook(make_hook("up_tfm_0")))

hooks.append(estimator.final_block.register_forward_hook(make_hook("final_block")))
hooks.append(estimator.final_proj.register_forward_hook(make_hook("final_proj")))

with torch.no_grad():
    output = estimator(x, mask, mu, t, spks, cond)

# Remove hooks
for h in hooks:
    h.remove()

# Add intermediates to outputs
outputs.update(intermediates)

save_file(outputs, "/outputs/cfm_estimator_steps.safetensors")
print(f"\nSaved to /outputs/cfm_estimator_steps.safetensors")
print(f"Keys: {list(outputs.keys())}")
