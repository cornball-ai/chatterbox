#!/usr/bin/env python3
"""Save full S3Gen pipeline reference for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import scipy.io.wavfile as wavfile

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen

# Load reference audio
print("\nLoading reference audio...")
ref_sr, ref_wav = wavfile.read("/scripts/reference.wav")
ref_wav = ref_wav.astype(np.float32) / 32768.0
print(f"Reference: {len(ref_wav)} samples at {ref_sr} Hz ({len(ref_wav)/ref_sr:.2f} sec)")

# Create same test tokens as R (seed 42)
print("\nCreating test tokens (same as R)...")
torch.manual_seed(42)
# Note: R uses runif which has different RNG from torch
# We'll save the actual tokens we use for exact comparison
test_tokens = torch.randint(0, 6560, (1, 50), device='cuda')
print(f"Test tokens: {test_tokens.shape}")

# Convert to torch
ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).float().cuda()

# Embed reference
print("\nEmbedding reference audio...")
with torch.inference_mode():
    ref_dict = s3gen.embed_ref(ref_wav_tensor, ref_sr)

print(f"  prompt_token: {ref_dict['prompt_token'].shape}")
print(f"  prompt_feat: {ref_dict['prompt_feat'].shape}")
print(f"  embedding: {ref_dict['embedding'].shape}")

# Run inference
print("\nRunning S3Gen inference...")
with torch.inference_mode():
    output_wav = s3gen.forward(
        speech_tokens=test_tokens,
        ref_wav=ref_wav_tensor,
        ref_sr=ref_sr
    )

print(f"\n=== Output ===")
print(f"  Shape: {output_wav.shape}")
print(f"  Mean: {output_wav.mean().item():.6f}")
print(f"  Std: {output_wav.std().item():.6f}")
print(f"  Range: [{output_wav.min().item():.4f}, {output_wav.max().item():.4f}]")
print(f"  Duration: {output_wav.shape[1]/24000:.2f} seconds")

# Save for comparison
save_dict = {
    "test_tokens": test_tokens.cpu().contiguous(),
    "prompt_token": ref_dict['prompt_token'].cpu().contiguous(),
    "prompt_feat": ref_dict['prompt_feat'].cpu().contiguous(),
    "embedding": ref_dict['embedding'].cpu().contiguous(),
    "output_wav": output_wav.cpu().contiguous(),
}
save_file(save_dict, "/outputs/s3gen_full_ref.safetensors")
print("\nSaved to /outputs/s3gen_full_ref.safetensors")

# Also save as WAV for listening
output_np = output_wav.squeeze().cpu().numpy()
output_int = (output_np * 32767).astype(np.int16)
wavfile.write("/outputs/s3gen_test_py.wav", 24000, output_int)
print("Saved audio to /outputs/s3gen_test_py.wav")
