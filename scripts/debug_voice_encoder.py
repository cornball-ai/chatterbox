#!/usr/bin/env python3
"""Debug voice encoder step by step."""

import torch
import safetensors.torch
import librosa
import numpy as np

from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.voice_encoder.melspec import melspectrogram
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load voice encoder
    ve_path = hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ve_path))
    ve.to(device).eval()

    # Load reference audio at 16kHz
    ref_audio = "/ref_audio/ShortCasey.wav"
    wav, sr = librosa.load(ref_audio, sr=16000)
    print(f"Original audio: {wav.shape}, sr={sr}")

    # Step 1: WITHOUT trimming (to compare mel directly)
    print("\n=== Without trimming ===")

    # Compute mel (Python style)
    hp = ve.hp
    mel = melspectrogram(wav, hp)
    print(f"Mel shape: {mel.shape}")
    print(f"Mel mean: {mel.mean():.6f}")
    print(f"Mel std: {mel.std():.6f}")
    print(f"Mel range: [{mel.min():.6f}, {mel.max():.6f}]")
    print(f"Mel first 5x5:\n{mel[:5, :5]}")

    # Step 2: WITH trimming (default behavior)
    print("\n=== With trimming (default) ===")
    wav_trimmed = librosa.effects.trim(wav, top_db=20)[0]
    print(f"Trimmed audio: {wav_trimmed.shape}")

    mel_trimmed = melspectrogram(wav_trimmed, hp)
    print(f"Mel shape: {mel_trimmed.shape}")
    print(f"Mel mean: {mel_trimmed.mean():.6f}")

    # Get embedding without trimming
    print("\n=== Embedding without trimming ===")
    with torch.no_grad():
        embed_no_trim = torch.from_numpy(
            ve.embeds_from_mels([mel.T])
        ).mean(axis=0, keepdim=True)
    print(f"Embedding mean: {embed_no_trim.mean().item():.6f}")
    print(f"Embedding std: {embed_no_trim.std().item():.6f}")

    # Get embedding with trimming (default)
    print("\n=== Embedding with trimming (default) ===")
    with torch.no_grad():
        embed_with_trim = torch.from_numpy(
            ve.embeds_from_wavs([wav], sample_rate=16000)
        ).mean(axis=0, keepdim=True)
    print(f"Embedding mean: {embed_with_trim.mean().item():.6f}")
    print(f"Embedding std: {embed_with_trim.std().item():.6f}")

    # Save outputs
    outputs = {
        "audio_wav": torch.from_numpy(wav).float(),
        "audio_wav_trimmed": torch.from_numpy(wav_trimmed).float(),
        "mel_no_trim": torch.from_numpy(mel.T).float(),  # (T, M)
        "mel_trimmed": torch.from_numpy(mel_trimmed.T).float(),
        "embedding_no_trim": embed_no_trim.float(),
        "embedding_trimmed": embed_with_trim.float(),
    }

    output_path = "/outputs/voice_encoder_debug.safetensors"
    safetensors.torch.save_file(outputs, output_path)
    print(f"\nSaved debug outputs to {output_path}")

if __name__ == "__main__":
    main()
