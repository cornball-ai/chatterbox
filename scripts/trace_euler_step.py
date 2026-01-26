#!/usr/bin/env python3
"""Trace single Euler step for comparison."""

import torch
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
decoder = model.s3gen.flow.decoder
decoder.eval()

torch.manual_seed(42)

with torch.no_grad():
    batch = 1
    time = 50

    mu = torch.randn(batch, 80, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    spks = torch.randn(batch, 80, device='cuda')
    cond = torch.randn(batch, 80, time, device='cuda')

    outputs = {
        "mu": mu.cpu().contiguous(),
        "mask": mask.cpu().contiguous(),
        "spks": spks.cpu().contiguous(),
        "cond": cond.cpu().contiguous(),
    }

    # Get initial noise
    z = decoder.rand_noise[:, :, :time].to(device='cuda')
    outputs["z_initial"] = z.cpu().contiguous()
    print(f"Initial noise: mean={z.mean().item():.6f}, std={z.std().item():.6f}")

    # Time schedule
    n_timesteps = 10
    t_span = torch.linspace(0, 1, n_timesteps + 1, device='cuda')
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    outputs["t_span"] = t_span.cpu().contiguous()
    print(f"t_span: {t_span.cpu().numpy()}")

    t, dt = t_span[0], t_span[1] - t_span[0]
    t = t.unsqueeze(0)

    # Setup CFG inputs
    x_in = torch.zeros([2, 80, time], device='cuda', dtype=z.dtype)
    mask_in = torch.zeros([2, 1, time], device='cuda', dtype=z.dtype)
    mu_in = torch.zeros([2, 80, time], device='cuda', dtype=z.dtype)
    t_in = torch.zeros([2], device='cuda', dtype=z.dtype)
    spks_in = torch.zeros([2, 80], device='cuda', dtype=z.dtype)
    cond_in = torch.zeros([2, 80, time], device='cuda', dtype=z.dtype)

    x = z.clone()

    # First step
    x_in[:] = x
    mask_in[:] = mask
    mu_in[0] = mu
    t_in[:] = t.unsqueeze(0)
    spks_in[0] = spks
    cond_in[0] = cond

    outputs["step0_x_in"] = x_in.cpu().contiguous()
    outputs["step0_mu_in"] = mu_in.cpu().contiguous()
    outputs["step0_t_in"] = t_in.cpu().contiguous()
    outputs["step0_spks_in"] = spks_in.cpu().contiguous()
    outputs["step0_cond_in"] = cond_in.cpu().contiguous()

    print(f"\nStep 0:")
    print(f"  t: {t.item():.6f}, dt: {dt.item():.6f}")
    print(f"  x_in: mean={x_in.mean().item():.6f}")
    print(f"  mu_in: mean={mu_in.mean().item():.6f} (batch 0), {mu_in[0].mean().item():.6f}")
    print(f"  spks_in: mean={spks_in.mean().item():.6f}")

    # Forward through estimator
    dphi_dt = decoder.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
    outputs["step0_dphi_raw"] = dphi_dt.cpu().contiguous()
    print(f"  dphi_dt (raw): mean={dphi_dt.mean().item():.6f}, std={dphi_dt.std().item():.6f}")

    # CFG combination
    dphi_cond = dphi_dt[0:1]
    dphi_uncond = dphi_dt[1:2]
    outputs["step0_dphi_cond"] = dphi_cond.cpu().contiguous()
    outputs["step0_dphi_uncond"] = dphi_uncond.cpu().contiguous()
    print(f"  dphi_cond: mean={dphi_cond.mean().item():.6f}")
    print(f"  dphi_uncond: mean={dphi_uncond.mean().item():.6f}")

    cfg_rate = decoder.inference_cfg_rate
    dphi_combined = (1.0 + cfg_rate) * dphi_cond - cfg_rate * dphi_uncond
    outputs["step0_dphi_combined"] = dphi_combined.cpu().contiguous()
    print(f"  cfg_rate: {cfg_rate}")
    print(f"  dphi_combined: mean={dphi_combined.mean().item():.6f}")

    # Euler step
    x = x + dt * dphi_combined
    outputs["step1_x"] = x.cpu().contiguous()
    print(f"  x after step: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

    # Run full decoder for comparison
    full_result, _ = decoder(mu=mu, mask=mask, spks=spks, cond=cond, n_timesteps=10)
    outputs["full_result"] = full_result.cpu().contiguous()
    print(f"\nFull result: mean={full_result.mean().item():.6f}, std={full_result.std().item():.6f}")

save_file(outputs, "/outputs/euler_step_trace.safetensors")
print("\nSaved to euler_step_trace.safetensors")
