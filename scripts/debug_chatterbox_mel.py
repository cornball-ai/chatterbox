#!/usr/bin/env python3
"""Debug what mel filterbank chatterbox actually uses."""

import numpy as np
from chatterbox.models.voice_encoder.melspec import mel_basis
from chatterbox.tts import ChatterboxTTS

# Load model to get hp
model = ChatterboxTTS.from_pretrained("cuda")
hp = model.ve.hp

print(f"Chatterbox VE hyperparameters:")
print(f"  sample_rate: {hp.sample_rate}")
print(f"  n_fft: {hp.n_fft}")
print(f"  num_mels: {hp.num_mels}")
print(f"  fmin: {hp.fmin}")
print(f"  fmax: {hp.fmax}")

# Get the actual mel basis used by chatterbox
fb = mel_basis(hp)
print(f"\nChatterbox mel_basis shape: {fb.shape}")

print("\nChatterbox filterbank bin sums:")
for i in [0, 9, 19, 29, 39]:
    print(f"  Bin {i+1}: sum={fb[i].sum():.4f}, max={fb[i].max():.4f}")

# Look at the mel_basis function source
import inspect
from chatterbox.models.voice_encoder import melspec
print("\n=== mel_basis source ===")
print(inspect.getsource(melspec.mel_basis))
