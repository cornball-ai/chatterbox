#!/usr/bin/env python3
"""Save estimator output with Euler step inputs for comparison."""

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

print(f"Inputs:")
print(f"  x_in: {x_in.shape}, mean={x_in.mean().item():.6f}")
print(f"  mu_in: {mu_in.shape}, mean={mu_in.mean().item():.6f}")
print(f"  t_in: {t_in.shape}, values={t_in.cpu().numpy()}")
print(f"  spks_in: {spks_in.shape}, mean={spks_in.mean().item():.6f}")
print(f"  cond_in: {cond_in.shape}, mean={cond_in.mean().item():.6f}")

# Forward
with torch.no_grad():
    output = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
    print(f"\nOutput: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
    print(f"Batch 0: mean={output[0].mean().item():.6f}")
    print(f"Batch 1: mean={output[1].mean().item():.6f}")

    # Compare with saved dphi_raw
    dphi_raw = euler_ref["step0_dphi_raw"].cuda()
    diff = (output - dphi_raw).abs().max().item()
    print(f"\nDiff vs saved dphi_raw: {diff:.6f}")

    # Save the new reference
    outputs = {
        "x_in": x_in.cpu().contiguous(),
        "mask_in": mask_in.cpu().contiguous(),
        "mu_in": mu_in.cpu().contiguous(),
        "t_in": t_in.cpu().contiguous(),
        "spks_in": spks_in.cpu().contiguous(),
        "cond_in": cond_in.cpu().contiguous(),
        "output": output.cpu().contiguous(),
    }
    save_file(outputs, "/outputs/estimator_euler_ref.safetensors")
    print(f"\nSaved to estimator_euler_ref.safetensors")
