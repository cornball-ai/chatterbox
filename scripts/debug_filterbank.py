#!/usr/bin/env python3
"""Debug mel filterbank normalization."""

import numpy as np
import librosa

# Create filterbank with and without normalization
fb_slaney = librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000, norm="slaney")
fb_none = librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000, norm=None)

print("With Slaney normalization:")
for i in [0, 9, 19, 29, 39]:
    print(f"  Bin {i+1}: sum={fb_slaney[i].sum():.4f}, max={fb_slaney[i].max():.4f}")

print("\nWithout normalization:")
for i in [0, 9, 19, 29, 39]:
    print(f"  Bin {i+1}: sum={fb_none[i].sum():.4f}, max={fb_none[i].max():.4f}")

# Check specific bin values
print("\nBin 1 (first 20 values) with slaney norm:")
print(f"  {fb_slaney[0, :20]}")

print("\nBin 1 (first 20 values) without norm:")
print(f"  {fb_none[0, :20]}")
