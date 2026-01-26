#!/usr/bin/env python3
"""Extract mel spectrogram from reference audio and save for R comparison."""

import torch
import torchaudio
import numpy as np
from safetensors.torch import save_file
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.voice_encoder.melspec import melspectrogram

# Load audio
audio_path = "/ref_audio/jfk.wav"
wav, sr = torchaudio.load(audio_path)
print(f"Audio shape: {wav.shape}, sample rate: {sr}")

# Convert stereo to mono if needed
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
    print(f"Converted to mono: {wav.shape}")

# Resample if needed (voice encoder expects 16kHz)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000
    print(f"Resampled to 16kHz: {wav.shape}")

# Load model
model = ChatterboxTTS.from_pretrained("cuda")

wav_np = wav.squeeze(0).numpy()

# Get hyperparameters
hp = model.ve.hp
print(f"\nVoice encoder hp attributes:")
for attr in dir(hp):
    if not attr.startswith('_'):
        val = getattr(hp, attr)
        if not callable(val):
            print(f"  {attr}: {val}")

# Compute mel using the VE's melspectrogram function
mel = melspectrogram(wav_np, hp)
print(f"\nMel from VE melspectrogram:")
print(f"  Shape: {mel.shape}")
print(f"  Mean: {mel.mean():.6f}")
print(f"  Std: {mel.std():.6f}")
print(f"  Min: {mel.min():.6f}")
print(f"  Max: {mel.max():.6f}")

# Get speaker embedding
with torch.inference_mode():
    speaker_emb = model.ve.embeds_from_wavs([wav_np], sample_rate=16000)
    print(f"\nSpeaker embedding shape: {speaker_emb.shape}")
    print(f"Speaker embedding mean: {speaker_emb.mean():.6f}")
    print(f"Speaker embedding std: {speaker_emb.std():.6f}")

# Save everything
save_file({
    "audio_wav": torch.from_numpy(wav_np),
    "mel": torch.from_numpy(mel),  # Shape: (n_mels, time) = (40, T)
    "speaker_embedding": torch.from_numpy(speaker_emb),
}, "/outputs/mel_reference.safetensors")

print(f"\nSaved to /outputs/mel_reference.safetensors")
print(f"  audio_wav: {wav_np.shape}")
print(f"  mel: {mel.shape}")
print(f"  speaker_embedding: {speaker_emb.shape}")
