#!/usr/bin/env python3
"""Trace how Python creates the PE buffer."""

import torch
from chatterbox import ChatterboxTTS

# Load model
print("Loading model...")
tts = ChatterboxTTS.from_pretrained('cuda')
pos_enc = tts.s3gen.flow.encoder.embed.pos_enc

print(f"pos_enc class: {pos_enc.__class__.__name__}")
print(f"pos_enc.xscale: {pos_enc.xscale}")

# Check the PE buffer
pe = pos_enc.pe
print(f"\nPE buffer shape: {pe.shape}")
print(f"PE buffer device: {pe.device}")

# Find where position 0 is (sin=0, cos=1)
# Check several positions
max_len = pe.shape[1]
center = max_len // 2

print(f"\nmax_len={max_len}, center={center}")

# Check positions around center
for offset in [-5, -1, 0, 1, 5]:
    idx = center + offset
    vals = pe[0, idx, :5].tolist()
    sin_vals = [vals[0], vals[2], vals[4]] if len(vals) >= 5 else vals[::2]
    cos_vals = [vals[1], vals[3]] if len(vals) >= 4 else vals[1::2]
    print(f"pe[{idx}] (offset {offset:+d}): sin={sin_vals[:3]}, cos={cos_vals[:2]}")

# Check if ESPnet creates positions from -(max_len-1) to (max_len-1)
# If so, position 0 is at index center

# Let's verify by checking the formula
# For position p, sin values should be sin(p * div_term)
# At center, if position is 0: sin(0) = 0, cos(0) = 1
# At center+1, if position is 1: sin(div_term[0]) = sin(1 * exp(0)) = sin(1) ≈ 0.841

import math
div_term_0 = math.exp(0 * (-math.log(10000.0) / 512))
print(f"\ndiv_term[0] = {div_term_0}")
print(f"sin(1 * div_term[0]) = {math.sin(1 * div_term_0):.6f}")
print(f"sin(-1 * div_term[0]) = {math.sin(-1 * div_term_0):.6f}")

# Check the actual source code for PE creation
import inspect
if hasattr(pos_enc, 'extend_pe'):
    print("\n--- extend_pe method ---")
    print(inspect.getsource(pos_enc.extend_pe))
