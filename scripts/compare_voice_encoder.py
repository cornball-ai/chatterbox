#!/usr/bin/env python3
"""Extract voice encoder reference for comparison with R implementation."""

import torch
import safetensors.torch
import librosa
import numpy as np

from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.s3tokenizer import S3_SR
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load voice encoder
    print("Loading voice encoder...")
    ve_path = hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ve_path))
    ve.to(device).eval()

    # Load reference audio
    ref_audio = "/ref_audio/ShortCasey.wav"
    print(f"Loading audio: {ref_audio}")

    # Load at 16kHz (S3_SR)
    wav, sr = librosa.load(ref_audio, sr=S3_SR)
    print(f"Audio shape: {wav.shape}, sample rate: {sr}")
    print(f"Audio duration: {len(wav)/sr:.2f}s")
    print(f"Audio range: [{wav.min():.4f}, {wav.max():.4f}]")

    # Get embedding
    print("\nComputing speaker embedding...")
    with torch.no_grad():
        embedding = torch.from_numpy(ve.embeds_from_wavs([wav], sample_rate=S3_SR))
        embedding = embedding.mean(axis=0, keepdim=True)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding mean: {embedding.mean().item():.6f}")
    print(f"Embedding std: {embedding.std().item():.6f}")
    print(f"Embedding range: [{embedding.min().item():.6f}, {embedding.max().item():.6f}]")

    # Save intermediate values for debugging
    outputs = {
        "audio_wav": torch.from_numpy(wav).float(),
        "speaker_embedding": embedding.float(),
    }

    # Note: mel_spectrogram is internal to voice encoder, skip for now

    output_path = "/outputs/voice_encoder_reference.safetensors"
    safetensors.torch.save_file(outputs, output_path)
    print(f"\nSaved reference to {output_path}")

if __name__ == "__main__":
    main()
