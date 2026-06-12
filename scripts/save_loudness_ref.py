#!/usr/bin/env python3
"""pyloudnorm reference values for R integrated_loudness validation."""

import numpy as np
import soundfile as sf
import pyloudnorm as ln

def meas(x, sr):
    return ln.Meter(sr).integrated_loudness(x)

sr = 24000
t = np.arange(5 * sr) / sr
sine1k = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
sine100 = 0.1 * np.sin(2 * np.pi * 100.0 * t)
mix = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 2500.0 * t)
# amplitude-modulated: quiet stretches exercise the gating
mod = (0.5 * np.sin(2 * np.pi * 0.3 * t) ** 2) * np.sin(2 * np.pi * 500.0 * t)

print(f"sine1k_24k  {meas(sine1k, sr):.6f}")
print(f"sine100_24k {meas(sine100, sr):.6f}")
print(f"mix_24k     {meas(mix, sr):.6f}")
print(f"mod_24k     {meas(mod, sr):.6f}")

# librosa-style load: chatterbox feeds the meter mono (channel mean)
wav, wsr = sf.read("/pkg/inst/audio/jfk.wav")
if wav.ndim == 2:
    wav = wav.mean(axis=1)
print(f"jfk_sr      {wsr}")
print(f"jfk_mono    {meas(wav, wsr):.6f}")
norm = wav * 10 ** ((-27 - meas(wav, wsr)) / 20)
print(f"jfk_norm27  {meas(norm, wsr):.6f}")
