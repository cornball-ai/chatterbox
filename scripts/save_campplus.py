#!/usr/bin/env python3
"""Save CAMPPlus speaker encoder steps for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen
speaker_encoder = s3gen.speaker_encoder

# Load reference audio
print("\nLoading reference audio...")
ref = load_file("/outputs/mel_reference.safetensors")
wav_np = ref["audio_wav"].numpy()
print(f"Reference audio: {len(wav_np)} samples")

# ============================================================================
# Step 1: CAMPPlus architecture
# ============================================================================
print("\n=== Step 1: CAMPPlus Architecture ===")
print(f"CAMPPlus type: {type(speaker_encoder).__name__}")

# Get inference method source
try:
    inf_source = inspect.getsource(speaker_encoder.inference)
    print("\nCAMPPlus.inference source:")
    print(inf_source)
except Exception as e:
    print(f"Error: {e}")

# Get forward method
try:
    fwd_source = inspect.getsource(type(speaker_encoder).forward)
    print("\nCAMPPlus.forward source:")
    print(fwd_source)
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Step 2: mel_extractor
# ============================================================================
print("\n=== Step 2: mel_extractor ===")
mel_extractor = s3gen.mel_extractor
print(f"mel_extractor type: {type(mel_extractor).__name__}")

try:
    mel_source = inspect.getsource(type(mel_extractor).forward)
    print("\nmel_extractor.forward source (first 50 lines):")
    lines = mel_source.split('\n')[:50]
    for line in lines:
        print(f"  {line}")
except Exception as e:
    print(f"Error: {e}")

# Check mel_extractor config
print("\nmel_extractor attributes:")
for attr in ['n_fft', 'hop_length', 'win_length', 'n_mels', 'sample_rate', 'f_min', 'f_max']:
    if hasattr(mel_extractor, attr):
        print(f"  {attr}: {getattr(mel_extractor, attr)}")

# ============================================================================
# Step 3: Run CAMPPlus step by step
# ============================================================================
print("\n=== Step 3: CAMPPlus Step by Step ===")
ref_wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).float().cuda()
print(f"ref_wav shape: {ref_wav_tensor.shape}")

with torch.inference_mode():
    # Step 3a: Compute mel spectrogram for CAMPPlus (different from output mel?)
    # Check what speaker_encoder.inference does internally

    # First compute mel at 24kHz for reference
    ref_mels_24 = mel_extractor(ref_wav_tensor).transpose(1, 2)
    print(f"\n3a. ref_mels_24 shape: {ref_mels_24.shape}")

    # Now call speaker_encoder.inference
    print("\n3b. Calling speaker_encoder.inference:")
    embedding = speaker_encoder.inference(ref_wav_tensor)
    print(f"embedding shape: {embedding.shape}")
    print(f"embedding mean: {embedding.mean().item():.6f}, std: {embedding.std().item():.6f}")

# ============================================================================
# Step 4: FCM head details
# ============================================================================
print("\n=== Step 4: FCM Head Details ===")
fcm = speaker_encoder.head
print(f"FCM type: {type(fcm).__name__}")
print("\nFCM children:")
for name, child in fcm.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'in_channels'):
        print(f"    in_channels: {child.in_channels}, out_channels: {child.out_channels}")

try:
    fcm_source = inspect.getsource(type(fcm).forward)
    print("\nFCM.forward source:")
    print(fcm_source)
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Step 5: xvector details
# ============================================================================
print("\n=== Step 5: xvector Details ===")
xvector = speaker_encoder.xvector
print("\nxvector children:")
for name, child in xvector.named_children():
    print(f"  {name}: {type(child).__name__}")

# ============================================================================
# Step 6: Save reference data
# ============================================================================
print("\n=== Step 6: Save Reference Data ===")
save_dict = {
    "ref_wav": ref_wav_tensor.cpu().contiguous(),
    "ref_mels_24": ref_mels_24.cpu().contiguous(),
    "embedding": embedding.cpu().contiguous(),
}
save_file(save_dict, "/outputs/campplus_reference.safetensors")
print("Saved to /outputs/campplus_reference.safetensors")
