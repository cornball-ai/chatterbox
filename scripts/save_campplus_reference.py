#!/usr/bin/env python3
"""Save CAMPPlus speaker encoder reference outputs for R validation."""

import torch
import torchaudio
from safetensors.torch import save_file
from chatterbox import ChatterboxTTS

# Load model (includes S3Gen with CAMPPlus)
print("Loading ChatterboxTTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device)

# Get CAMPPlus speaker encoder from s3gen
campplus = tts.s3gen.speaker_encoder
campplus.eval()
print(f"CAMPPlus type: {type(campplus)}")

# Load reference audio
print("Loading reference audio...")
ref_path = "/scripts/reference.wav"
wav, sr = torchaudio.load(ref_path)
print(f"Loaded audio: shape={wav.shape}, sr={sr}")

# Convert to mono if stereo
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)

# Resample to 16kHz for CAMPPlus
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wav_16k = resampler(wav)
else:
    wav_16k = wav

wav_16k = wav_16k.to(device)
print(f"Audio shape: {wav_16k.shape}, sample rate: 16000")

# Extract fbank features (what CAMPPlus expects)
print("Extracting fbank features...")
with torch.no_grad():
    # CAMPPlus uses torchaudio Kaldi fbank
    fbank = torchaudio.compliance.kaldi.fbank(
        wav_16k,
        num_mel_bins=80,
        sample_frequency=16000,
        frame_length=25.0,
        frame_shift=10.0,
    )
    # Normalize by subtracting mean
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    fbank = fbank.unsqueeze(0)  # Add batch dim: (1, T, 80)
    print(f"Fbank shape: {fbank.shape}")

# Run through CAMPPlus step by step
print("Running CAMPPlus forward pass...")
with torch.no_grad():
    # Input: (B, T, F) -> (B, F, T)
    x = fbank.permute(0, 2, 1)
    print(f"After permute: {x.shape}")

    # FCM head
    x_fcm = campplus.head(x)
    print(f"After FCM head: {x_fcm.shape}")

    # TDNN - note: structure is campplus.xvector.tdnn
    x_tdnn = campplus.xvector.tdnn(x_fcm)
    print(f"After TDNN: {x_tdnn.shape}")

    # Block 1
    x_block1 = campplus.xvector.block1(x_tdnn)
    x_transit1 = campplus.xvector.transit1(x_block1)
    print(f"After block1+transit1: {x_transit1.shape}")

    # Block 2
    x_block2 = campplus.xvector.block2(x_transit1)
    x_transit2 = campplus.xvector.transit2(x_block2)
    print(f"After block2+transit2: {x_transit2.shape}")

    # Block 3
    x_block3 = campplus.xvector.block3(x_transit2)
    x_transit3 = campplus.xvector.transit3(x_block3)
    print(f"After block3+transit3: {x_transit3.shape}")

    # Output nonlinear
    x_out = torch.relu(campplus.xvector.out_nonlinear(x_transit3))
    print(f"After out_nonlinear: {x_out.shape}")

    # Stats pooling
    mean_x = x_out.mean(dim=2)
    std_x = x_out.std(dim=2)
    x_stats = torch.cat([mean_x, std_x], dim=1)
    print(f"After stats pooling: {x_stats.shape}")

    # Dense layer
    embedding = campplus.xvector.dense(x_stats)
    print(f"Final embedding: {embedding.shape}")

    # Also get full forward pass result
    full_embedding = campplus(fbank)
    print(f"Full forward embedding: {full_embedding.shape}")
    print(f"Embeddings match: {torch.allclose(embedding, full_embedding, atol=1e-5)}")

# Save all intermediates
print("Saving outputs...")
outputs = {
    "fbank": fbank.cpu().float().contiguous(),
    "after_permute": fbank.permute(0, 2, 1).cpu().float().contiguous(),
    "after_fcm": x_fcm.cpu().float().contiguous(),
    "after_tdnn": x_tdnn.cpu().float().contiguous(),
    "after_block1": x_block1.cpu().float().contiguous(),
    "after_transit1": x_transit1.cpu().float().contiguous(),
    "after_block2": x_block2.cpu().float().contiguous(),
    "after_transit2": x_transit2.cpu().float().contiguous(),
    "after_block3": x_block3.cpu().float().contiguous(),
    "after_transit3": x_transit3.cpu().float().contiguous(),
    "after_out_nonlinear": x_out.cpu().float().contiguous(),
    "stats_pooled": x_stats.cpu().float().contiguous(),
    "embedding": embedding.cpu().float().contiguous(),
    "full_embedding": full_embedding.cpu().float().contiguous(),
}

save_file(outputs, "/outputs/campplus_reference.safetensors")
print("Saved to /outputs/campplus_reference.safetensors")

# Print statistics for comparison
print("\n=== Statistics for comparison ===")
print(f"Fbank - mean: {fbank.mean().item():.6f}, std: {fbank.std().item():.6f}")
print(f"FCM output - mean: {x_fcm.mean().item():.6f}, std: {x_fcm.std().item():.6f}")
print(f"TDNN output - mean: {x_tdnn.mean().item():.6f}, std: {x_tdnn.std().item():.6f}")
print(f"Transit1 - mean: {x_transit1.mean().item():.6f}, std: {x_transit1.std().item():.6f}")
print(f"Transit2 - mean: {x_transit2.mean().item():.6f}, std: {x_transit2.std().item():.6f}")
print(f"Transit3 - mean: {x_transit3.mean().item():.6f}, std: {x_transit3.std().item():.6f}")
print(f"Embedding - mean: {embedding.mean().item():.6f}, std: {embedding.std().item():.6f}")
print(f"Embedding L2 norm: {torch.norm(embedding).item():.6f}")
