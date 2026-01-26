#!/usr/bin/env python3
"""Extract mel filterbank and STFT params from Python for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS
import librosa

# Load model to get hyperparameters
model = ChatterboxTTS.from_pretrained("cuda")
hp = model.ve.hp

print("Voice encoder hyperparameters:")
print(f"  sample_rate: {hp.sample_rate}")
print(f"  n_fft: {hp.n_fft}")
print(f"  win_size: {hp.win_size}")
print(f"  hop_size: {hp.hop_size}")
print(f"  num_mels: {hp.num_mels}")
print(f"  fmin: {hp.fmin}")
print(f"  fmax: {hp.fmax}")
print(f"  mel_type: {hp.mel_type}")
print(f"  mel_power: {hp.mel_power}")
print(f"  preemphasis: {hp.preemphasis}")
print(f"  stft_magnitude_min: {hp.stft_magnitude_min}")
print(f"  normalized_mels: {hp.normalized_mels}")

# Create mel filterbank using librosa (what Python chatterbox uses)
mel_fb = librosa.filters.mel(
    sr=hp.sample_rate,
    n_fft=hp.n_fft,
    n_mels=hp.num_mels,
    fmin=hp.fmin,
    fmax=hp.fmax,
    htk=False,  # Use Slaney formula by default
    norm="slaney"
)

print(f"\nMel filterbank shape: {mel_fb.shape}")
print(f"Mel filterbank sum per bin: {mel_fb.sum(axis=1)[:5]}...")

# Show the frequency ranges for each mel bin
mel_freqs = librosa.mel_frequencies(n_mels=hp.num_mels + 2, fmin=hp.fmin, fmax=hp.fmax)
print(f"\nMel center frequencies (first 10):")
for i in range(min(10, len(mel_freqs))):
    print(f"  {i}: {mel_freqs[i]:.1f} Hz")

# Also check the STFT parameters
print(f"\nSTFT frequency bins: {hp.n_fft // 2 + 1}")
fft_freqs = librosa.fft_frequencies(sr=hp.sample_rate, n_fft=hp.n_fft)
print(f"FFT frequencies (first 10): {fft_freqs[:10]}")

# Look at the actual melspectrogram function
from chatterbox.models.voice_encoder.melspec import melspectrogram
import inspect
print("\n=== melspectrogram source ===")
source = inspect.getsource(melspectrogram)
print(source)

# Save filterbank for R comparison
save_file({
    "mel_filterbank": torch.from_numpy(mel_fb.astype(np.float32)),
    "fft_freqs": torch.from_numpy(fft_freqs.astype(np.float32)),
    "mel_freqs": torch.from_numpy(mel_freqs.astype(np.float32)),
}, "/outputs/filterbank_reference.safetensors")

print("\nSaved filterbank to /outputs/filterbank_reference.safetensors")
