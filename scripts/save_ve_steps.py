#!/usr/bin/env python3
"""Save voice encoder computation step by step for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.voice_encoder.melspec import melspectrogram

# Load the model
print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
ve = model.ve
hp = ve.hp

print(f"Voice encoder config:")
print(f"  ve_partial_frames: {hp.ve_partial_frames}")
print(f"  speaker_embed_size: {hp.speaker_embed_size}")
print(f"  ve_hidden_size: {hp.ve_hidden_size}")

# Load the same audio as mel_reference
ref = load_file("/outputs/mel_reference.safetensors")
audio_wav = ref["audio_wav"].numpy()
print(f"\nAudio: {len(audio_wav)} samples")

# Step 1: Compute mel spectrogram (same as save_mel_steps.py)
mel = melspectrogram(audio_wav, hp, pad=True)
print(f"\n1. Mel spectrogram shape: {mel.shape}")  # (n_mels, time)

# Step 2: Transpose for LSTM input (time, mels)
mel_t = mel.T  # (time, mels)
print(f"2. Transposed mel shape: {mel_t.shape}")

# Step 3: Split into overlapping partials
# Python uses ve.compute_partial_slices()
n_frames = mel_t.shape[0]
partial_frames = hp.ve_partial_frames  # 160
overlap = 0.5
frame_step = int(round(partial_frames * (1 - overlap)))
print(f"\n3. Partial computation:")
print(f"   n_frames: {n_frames}")
print(f"   partial_frames: {partial_frames}")
print(f"   frame_step: {frame_step}")

# Compute number of partials
n_partials = (n_frames - partial_frames + frame_step) // frame_step
if n_partials == 0:
    n_partials = 1
print(f"   n_partials: {n_partials}")

# Extract partials
partials = []
for i in range(n_partials):
    start = i * frame_step
    end = start + partial_frames
    if end > n_frames:
        # Pad if needed
        partial = np.zeros((partial_frames, hp.num_mels))
        actual_len = n_frames - start
        partial[:actual_len] = mel_t[start:n_frames]
    else:
        partial = mel_t[start:end]
    partials.append(partial)
    if i < 3:  # Print first few
        print(f"   partial {i}: frames {start}-{end}, shape {partial.shape}")

partials_np = np.stack(partials, axis=0)  # (n_partials, partial_frames, n_mels)
print(f"\n4. Partials tensor shape: {partials_np.shape}")

# Convert to torch and run through model
partials_t = torch.from_numpy(partials_np.astype(np.float32)).cuda()

# Step 5: Run through LSTM
with torch.no_grad():
    lstm_out, (hidden, cell) = ve.lstm(partials_t)
    print(f"\n5. LSTM output:")
    print(f"   lstm_out shape: {lstm_out.shape}")
    print(f"   hidden shape: {hidden.shape}")  # (num_layers, batch, hidden)

    # Get final layer hidden state
    final_hidden = hidden[-1]  # (batch, hidden_size)
    print(f"   final_hidden shape: {final_hidden.shape}")

    # Step 6: Project to embedding
    raw_embeds = ve.proj(final_hidden)
    print(f"\n6. Raw projection shape: {raw_embeds.shape}")
    print(f"   Raw projection mean: {raw_embeds.mean().item():.6f}")

    # Step 7: Apply ReLU
    relu_embeds = torch.relu(raw_embeds)
    print(f"\n7. After ReLU mean: {relu_embeds.mean().item():.6f}")

    # Step 8: L2 normalize each partial
    partial_embeds = relu_embeds / torch.norm(relu_embeds, dim=1, keepdim=True)
    print(f"\n8. Normalized partial embeds shape: {partial_embeds.shape}")
    print(f"   First partial embed (first 10): {partial_embeds[0, :10].cpu().numpy()}")

    # Step 9: Average partials
    mean_embed = partial_embeds.mean(dim=0, keepdim=True)
    print(f"\n9. Mean embed (before final norm): {mean_embed[0, :10].cpu().numpy()}")

    # Step 10: Final L2 normalize
    speaker_embed = mean_embed / torch.norm(mean_embed, dim=1, keepdim=True)
    print(f"\n10. Final speaker embedding:")
    print(f"    shape: {speaker_embed.shape}")
    print(f"    mean: {speaker_embed.mean().item():.6f}")
    print(f"    first 10: {speaker_embed[0, :10].cpu().numpy()}")

# Also run through the official interface for comparison
print("\n=== Official interface ===")
with torch.no_grad():
    official_embed = ve.embeds_from_wavs([audio_wav], sample_rate=16000)
    print(f"Official embed shape: {official_embed.shape}")
    print(f"Official embed mean: {official_embed.mean():.6f}")
    print(f"Official first 10: {official_embed[0, :10]}")

    # Check if they match
    official_t = torch.from_numpy(official_embed).cuda()
    diff = (speaker_embed - official_t).abs()
    print(f"\nDiff vs official: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

# Save for R comparison
save_file({
    "mel_for_ve": torch.from_numpy(mel.T.astype(np.float32).copy()),  # (time, mels)
    "partials": torch.from_numpy(partials_np.astype(np.float32).copy()),
    "final_hidden": final_hidden.cpu().contiguous(),
    "raw_embeds": raw_embeds.cpu().contiguous(),
    "relu_embeds": relu_embeds.cpu().contiguous(),
    "partial_embeds": partial_embeds.cpu().contiguous(),
    "speaker_embedding": speaker_embed.cpu().contiguous(),
    "official_embedding": torch.from_numpy(official_embed.astype(np.float32).copy()),
}, "/outputs/ve_steps.safetensors")

print("\nSaved to /outputs/ve_steps.safetensors")
