#!/usr/bin/env python3
"""Run Python ChatterboxVC for R voice_convert comparison."""

import glob
import torch
import soundfile as sf
from chatterbox.vc import ChatterboxVC

snap = glob.glob("/root/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*")[0]
vc = ChatterboxVC.from_local(snap, "cuda")
wav = vc.generate("/pkg/inst/audio/jfk.wav", target_voice_path="/pkg/scripts/reference.wav")
wav = wav.squeeze(0).numpy()
print(f"py vc: {len(wav)/24000:.2f}s, std={wav.std():.4f}")
sf.write("/outputs/vc_py_jfk_to_reference.wav", wav, 24000)
