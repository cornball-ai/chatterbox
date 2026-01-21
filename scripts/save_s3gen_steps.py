#!/usr/bin/env python3
"""Save S3Gen computation step by step for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import scipy.io.wavfile as wavfile

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen

# Save reference audio to a temp file
print("\nLoading reference audio...")
ref = load_file("/outputs/mel_reference.safetensors")
wav_np = ref["audio_wav"].numpy()
print(f"Reference audio: {len(wav_np)} samples")
wavfile.write("/tmp/ref_audio.wav", 16000, (wav_np * 32767).astype(np.int16))

# Get speaker embedding
print("\n=== Step 1: Speaker Embedding ===")
with torch.no_grad():
    speaker_emb = model.ve.embeds_from_wavs([wav_np], sample_rate=16000)
    speaker_emb = torch.from_numpy(speaker_emb).cuda()
print(f"Speaker embedding: {speaker_emb.shape}")

# Generate speech tokens using full pipeline
print("\n=== Step 2: Generate Speech Tokens (T3) ===")
text = "Hello world"
from chatterbox.tts import punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond

text_normalized = punc_norm(text)
text_tokens = model.tokenizer.text_to_tokens(text_normalized).to('cuda')
print(f"Text: '{text}' -> tokens: {text_tokens.shape}")

# Get prompt tokens
max_len = model.t3.hp.speech_cond_prompt_len
with torch.no_grad():
    prompt_tokens, _ = s3gen.tokenizer.forward([wav_np], max_len=max_len)
    prompt_tokens = torch.atleast_2d(prompt_tokens).cuda()
print(f"Prompt tokens: {prompt_tokens.shape}")

# Build T3Cond
t3_cond = T3Cond(
    speaker_emb=speaker_emb,
    cond_prompt_speech_tokens=prompt_tokens,
    emotion_adv=0.5 * torch.ones(1, 1, 1),
).to(device='cuda')

# Use full TTS to get both speech tokens and audio
torch.manual_seed(42)
with torch.no_grad():
    # Get speech tokens via internal method
    # First prepare input
    cond = model.t3.prepare_conditioning(t3_cond)

    # Manually run T3 sampling loop to get speech tokens
    # This is complex, so let's use the full generate and reverse-engineer

    # Actually, use model.generate which returns audio
    audio = model.generate(
        text=text,
        audio_prompt_path="/tmp/ref_audio.wav",
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )
    print(f"Generated audio: {audio.shape}")

# For now, let's trace S3Gen with synthetic tokens
print("\n=== Step 3: S3Gen Architecture ===")
print(f"S3Gen modules:")
for name, child in s3gen.named_children():
    print(f"  {name}: {type(child).__name__}")

# Check if we can get intermediate outputs from S3Gen
print("\n=== Step 4: S3Gen Flow ===")

# S3Gen needs speech_tokens and speaker_emb
# Let's create some test tokens and trace the flow
test_tokens = torch.randint(0, 6561, (1, 31), device='cuda')  # Random tokens
print(f"Test tokens: {test_tokens.shape}")

import inspect

with torch.inference_mode():
    # Check flow module
    print("Flow module:", type(s3gen.flow).__name__)
    sig = inspect.signature(s3gen.flow.forward)
    print(f"  forward signature: {sig}")

    # Check mel2wav (HiFTGenerator)
    print("\nHiFTGenerator module:", type(s3gen.mel2wav).__name__)
    sig = inspect.signature(s3gen.mel2wav.forward)
    print(f"  forward signature: {sig}")

    # Check the main forward method
    print("\nS3Gen forward:")
    sig = inspect.signature(s3gen.forward)
    print(f"  signature: {sig}")

    # Try running S3Gen forward with test tokens
    print("\nTrying S3Gen forward on test tokens...")
    # Convert ref audio to tensor
    ref_wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).float().cuda()
    print(f"Reference wav tensor: {ref_wav_tensor.shape}")

    try:
        s3gen_audio = s3gen.forward(
            speech_tokens=test_tokens,
            ref_wav=ref_wav_tensor,
            ref_sr=16000,
        )
        print(f"S3Gen output: {s3gen_audio.shape}")
        print(f"  mean={s3gen_audio.mean().item():.6f}, std={s3gen_audio.std().item():.6f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        s3gen_audio = None

# ============================================================================
# Save for R comparison
# ============================================================================
print("\n=== Saving Reference ===")

audio_np = audio.squeeze().cpu().numpy() if isinstance(audio, torch.Tensor) else audio

save_dict = {
    "speaker_emb": speaker_emb.cpu().contiguous(),
    "audio": torch.from_numpy(audio_np).unsqueeze(0).contiguous(),
    "text_tokens": text_tokens.cpu().contiguous(),
    "prompt_tokens": prompt_tokens.cpu().contiguous(),
    "test_tokens": test_tokens.cpu().contiguous(),
}

if s3gen_audio is not None:
    save_dict["s3gen_test_audio"] = s3gen_audio.cpu().contiguous()

save_file(save_dict, "/outputs/s3gen_steps.safetensors")
print("Saved to /outputs/s3gen_steps.safetensors")

# Also save audio as WAV
wavfile.write("/outputs/s3gen_test.wav", 24000, (audio_np * 32767).astype(np.int16))
print("Saved audio to /outputs/s3gen_test.wav")
