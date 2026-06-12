#!/usr/bin/env python3
"""Capture solve_euler inputs/output during a real generation.

Produces /outputs/euler_step_trace.safetensors with the keys
test_cfm_full.R expects: mu, mask, spks, cond, z_initial, full_result.
Regenerated June 2026 to re-validate the R CFM after its attention
moved to torch_scaled_dot_product_attention.
"""

import torch
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
decoder = model.s3gen.flow.decoder

captured = {}
orig = decoder.solve_euler


def hook(x, t_span, mu, mask, spks, cond, **kw):  # 0.1.7 adds meanflow=
    if not captured:  # first call only
        captured.update(dict(
            z_initial=x, t_span=t_span, mu=mu, mask=mask,
            spks=spks, cond=cond,
        ))
    out = orig(x, t_span, mu=mu, mask=mask, spks=spks, cond=cond, **kw)
    if "full_result" not in captured:
        captured["full_result"] = out[0] if isinstance(out, tuple) else out
    return out


decoder.solve_euler = hook

torch.manual_seed(42)
print("Generating...")
wav = model.generate(
    "The quick brown fox jumps over the lazy dog near the river bank.",
    audio_prompt_path="/scripts/reference.wav",
)
print(f"Audio: {wav.shape}, std={wav.float().std().item():.6f}")

save = {k: v.detach().float().contiguous().cpu()
        for k, v in captured.items()}
for k, v in save.items():
    print(f"  {k}: {list(v.shape)}")
save_file(save, "/outputs/euler_step_trace.safetensors")
print("Saved /outputs/euler_step_trace.safetensors")
