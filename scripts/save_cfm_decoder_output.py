#!/usr/bin/env python3
"""Save full CFM decoder output for R comparison."""

import torch
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
from einops import pack, repeat

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
decoder = model.s3gen.flow.decoder
decoder.eval()

# Use same random seed
torch.manual_seed(42)

with torch.no_grad():
    batch = 1
    time = 50

    # Create test inputs for decoder
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

    print(f"Inputs:")
    print(f"  mu: {mu.shape}, mean={mu.mean().item():.6f}")
    print(f"  mask: {mask.shape}")
    print(f"  spks: {spks.shape}, mean={spks.mean().item():.6f}")
    print(f"  cond: {cond.shape}, mean={cond.mean().item():.6f}")

    # Run decoder
    result, _ = decoder(
        mu=mu,
        mask=mask,
        spks=spks,
        cond=cond,
        n_timesteps=10
    )
    outputs["decoder_output"] = result.cpu().contiguous()
    print(f"\nDecoder output: shape={result.shape}, mean={result.mean().item():.6f}, std={result.std().item():.6f}")

    # Also save initial noise if available
    if hasattr(decoder, 'rand_noise'):
        rand_noise = decoder.rand_noise[:, :, :time]
        outputs["rand_noise"] = rand_noise.cpu().contiguous()
        print(f"rand_noise: shape={rand_noise.shape}, mean={rand_noise.mean().item():.6f}")

save_file(outputs, "/outputs/cfm_decoder_output.safetensors")
print(f"\nSaved to cfm_decoder_output.safetensors")
