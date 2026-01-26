#!/usr/bin/env python3
"""Compare STFT between librosa and what R is doing."""

import numpy as np
import torch
import torchaudio
import librosa
from safetensors.torch import save_file

# Load audio
wav, sr = torchaudio.load("/ref_audio/jfk.wav")
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
wav_np = wav.squeeze(0).numpy()

print(f"Audio shape: {wav_np.shape}, samples: {len(wav_np)}")

# STFT parameters
n_fft = 400
hop_size = 160
win_size = 400

# 1. librosa STFT (what Python chatterbox uses)
librosa_stft = librosa.stft(wav_np, n_fft=n_fft, hop_length=hop_size,
                             win_length=win_size, center=True, pad_mode="reflect")
librosa_mag = np.abs(librosa_stft)
print(f"\nLibrosa STFT:")
print(f"  Shape: {librosa_stft.shape}")
print(f"  Magnitude mean: {librosa_mag.mean():.6f}")
print(f"  Magnitude max: {librosa_mag.max():.6f}")

# 2. torch STFT with center=True (to match librosa)
hann_window = torch.hann_window(win_size)
torch_stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                        window=hann_window, center=True, pad_mode="reflect",
                        normalized=False, onesided=True, return_complex=True)
torch_mag = torch_stft.abs().squeeze(0).numpy()
print(f"\nTorch STFT (center=True):")
print(f"  Shape: {torch_mag.shape}")
print(f"  Magnitude mean: {torch_mag.mean():.6f}")
print(f"  Magnitude max: {torch_mag.max():.6f}")

# 3. torch STFT with center=False and manual padding (what R does)
pad_amount = n_fft // 2
wav_padded = torch.nn.functional.pad(wav.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)
torch_stft_manual = torch.stft(wav_padded, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                               window=hann_window, center=False, pad_mode="reflect",
                               normalized=False, onesided=True, return_complex=True)
torch_mag_manual = torch_stft_manual.abs().squeeze(0).numpy()
print(f"\nTorch STFT (center=False, manual padding):")
print(f"  Shape: {torch_mag_manual.shape}")
print(f"  Magnitude mean: {torch_mag_manual.mean():.6f}")
print(f"  Magnitude max: {torch_mag_manual.max():.6f}")

# Compare
diff_torch_center = np.abs(librosa_mag - torch_mag)
print(f"\nDiff (librosa vs torch center=True): max={diff_torch_center.max():.6f}, mean={diff_torch_center.mean():.6f}")

# Trim to same length for comparison
min_time = min(librosa_mag.shape[1], torch_mag_manual.shape[1])
diff_manual = np.abs(librosa_mag[:, :min_time] - torch_mag_manual[:, :min_time])
print(f"Diff (librosa vs torch manual pad): max={diff_manual.max():.6f}, mean={diff_manual.mean():.6f}")

# Check frame alignment
print(f"\nFrame 100 comparison (first 10 freq bins):")
print(f"  Librosa:       {librosa_mag[:10, 100]}")
print(f"  Torch center:  {torch_mag[:10, 100]}")
print(f"  Torch manual:  {torch_mag_manual[:10, 100]}")

# Save for R comparison
save_file({
    "librosa_mag": torch.from_numpy(librosa_mag.astype(np.float32)),
    "torch_mag_center": torch.from_numpy(torch_mag.astype(np.float32)),
    "torch_mag_manual": torch.from_numpy(torch_mag_manual.astype(np.float32)),
}, "/outputs/stft_reference.safetensors")
print("\nSaved to /outputs/stft_reference.safetensors")
