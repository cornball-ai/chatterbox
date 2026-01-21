#!/usr/bin/env python3
"""Save mel computation step by step for R comparison."""

import torch
import numpy as np
import torchaudio
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.voice_encoder.melspec import melspectrogram, mel_basis, _stft

# Load audio
wav, sr = torchaudio.load("/ref_audio/jfk.wav")
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
wav_np = wav.squeeze(0).numpy()

print(f"Audio: {len(wav_np)} samples")

# Load model to get hp
model = ChatterboxTTS.from_pretrained("cuda")
hp = model.ve.hp

# Step 1: STFT
stft_complex = _stft(wav_np, hp, pad=True)
print(f"STFT shape: {stft_complex.shape}")

# Step 2: Magnitude
magnitude = np.abs(stft_complex)
print(f"Magnitude shape: {magnitude.shape}")
print(f"Magnitude frame 100, bins 0-5: {magnitude[:5, 100]}")

# Step 3: Apply power
mag_pow = magnitude ** hp.mel_power
print(f"After power ({hp.mel_power}): {mag_pow[:5, 100]}")

# Step 4: Mel filterbank
fb = mel_basis(hp)
print(f"Mel filterbank shape: {fb.shape}")

# Step 5: Apply mel
mel = np.dot(fb, mag_pow)
print(f"Mel shape: {mel.shape}")
print(f"Mel frame 100, bins 15-25: {mel[15:25, 100]}")

# For comparison, also compute with melspectrogram function
mel_direct = melspectrogram(wav_np, hp, pad=True)
print(f"\nDirect melspectrogram shape: {mel_direct.shape}")
print(f"Direct mel frame 100, bins 15-25: {mel_direct[15:25, 100]}")

# Save everything for R comparison
# Note: saving as contiguous tensors
save_file({
    "stft_magnitude": torch.from_numpy(magnitude.astype(np.float32).copy()),
    "mag_pow": torch.from_numpy(mag_pow.astype(np.float32).copy()),
    "mel_filterbank": torch.from_numpy(fb.astype(np.float32).copy()),
    "mel": torch.from_numpy(mel.astype(np.float32).copy()),
}, "/outputs/mel_steps.safetensors")

print("\nSaved to /outputs/mel_steps.safetensors")
print(f"  stft_magnitude: {magnitude.shape}")
print(f"  mag_pow: {mag_pow.shape}")
print(f"  mel_filterbank: {fb.shape}")
print(f"  mel: {mel.shape}")
