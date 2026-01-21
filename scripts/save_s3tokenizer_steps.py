#!/usr/bin/env python3
"""Save S3 Tokenizer computation step by step for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3tokenizer import S3_SR

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")

# Load reference audio from mel_reference.safetensors (16kHz)
ref = load_file("/outputs/mel_reference.safetensors")
wav_16k = ref["audio_wav"].numpy()
print(f"\nAudio at 16kHz: {len(wav_16k)} samples ({len(wav_16k)/16000:.2f}s)")

# S3 Tokenizer uses 24kHz
print(f"\nS3 Tokenizer sample rate: {S3_SR} Hz")

# Resample to 24kHz
import librosa
wav_24k = librosa.resample(wav_16k, orig_sr=16000, target_sr=S3_SR)
print(f"Audio at 24kHz: {len(wav_24k)} samples ({len(wav_24k)/S3_SR:.2f}s)")

# Get the S3 tokenizer
s3_tokenizer = model.s3gen.tokenizer
print(f"\nS3 Tokenizer type: {type(s3_tokenizer).__name__}")

# Check tokenizer attributes
print("\nS3 Tokenizer attributes:")
for attr in dir(s3_tokenizer):
    if not attr.startswith('_'):
        val = getattr(s3_tokenizer, attr)
        if not callable(val) and not isinstance(val, torch.nn.Module):
            print(f"  {attr}: {val}")

# Tokenize
print("\n=== Tokenization ===")
with torch.no_grad():
    # Convert to tensor
    wav_tensor = torch.from_numpy(wav_24k).unsqueeze(0).cuda()
    print(f"Input tensor shape: {wav_tensor.shape}")

    # Get max_len from T3 config
    max_len = model.t3.hp.speech_cond_prompt_len
    print(f"Max prompt length: {max_len}")

    # Tokenize
    tokens, token_lens = s3_tokenizer.forward([wav_24k], max_len=max_len)
    print(f"\nTokens shape: {tokens.shape}")
    print(f"Token lens: {token_lens}")
    print(f"First 20 tokens: {tokens[0, :20].cpu().numpy() if torch.is_tensor(tokens) else tokens[0, :20]}")
    print(f"Last 20 tokens: {tokens[0, -20:].cpu().numpy() if torch.is_tensor(tokens) else tokens[0, -20:]}")

    # Get token statistics
    if torch.is_tensor(tokens):
        tokens_np = tokens.cpu().numpy()
    else:
        tokens_np = np.array(tokens)
    print(f"\nToken stats: min={tokens_np.min()}, max={tokens_np.max()}, unique={len(np.unique(tokens_np))}")

# Check what the tokenizer does internally
print("\n=== Tokenizer internals ===")
print(f"Tokenizer children:")
for name, child in s3_tokenizer.named_children():
    print(f"  {name}: {type(child).__name__}")

# Try to trace the mel spectrogram computation
print("\n=== Mel computation for S3 ===")
# Check if there's a mel extractor
if hasattr(s3_tokenizer, 'mel_extractor'):
    print(f"mel_extractor: {type(s3_tokenizer.mel_extractor).__name__}")
if hasattr(s3_tokenizer, 'feature_extractor'):
    print(f"feature_extractor: {type(s3_tokenizer.feature_extractor).__name__}")
    fe = s3_tokenizer.feature_extractor
    if hasattr(fe, 'mel_spec'):
        print(f"  mel_spec config: {fe.mel_spec}")

# Save reference data
save_dict = {
    "wav_16k": torch.from_numpy(wav_16k.astype(np.float32)),
    "wav_24k": torch.from_numpy(wav_24k.astype(np.float32)),
    "prompt_speech_tokens": torch.tensor(tokens_np) if not torch.is_tensor(tokens) else tokens.cpu(),
}

save_file(save_dict, "/outputs/s3tokenizer_steps.safetensors")
print("\nSaved to /outputs/s3tokenizer_steps.safetensors")
