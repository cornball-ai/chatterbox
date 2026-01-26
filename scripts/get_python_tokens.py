#!/usr/bin/env python3
"""Get Python's text tokenization"""

import sys
sys.path.insert(0, '/app/.venv/lib/python3.11/site-packages')

import torch
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")

text = "cornball AI is doing something for our country!"
ref_path = "/scripts/reference.wav"

# Find tokenizer - look at all attributes
print("\nT3 attributes with 'token' in name:")
for attr in dir(model.t3):
    if 'token' in attr.lower():
        print(f"  {attr}")

# Check hp
hp = model.t3.hp
print(f"\nConfig (hp):")
print(f"  start_text_token: {hp.start_text_token}")
print(f"  stop_text_token: {hp.stop_text_token}")
print(f"  start_speech_token: {hp.start_speech_token}")
print(f"  stop_speech_token: {hp.stop_speech_token}")
print(f"  speech_cond_prompt_len: {hp.speech_cond_prompt_len}")

# Look at methods
print("\nT3 callable methods:")
for attr in dir(model.t3):
    if not attr.startswith('_') and callable(getattr(model.t3, attr)):
        print(f"  {attr}")

# Try different ways to tokenize
print("\n\nTrying tokenization methods:")
for method_name in ['tokenize', 'tokenize_text', 'encode_text', 'text_to_tokens']:
    if hasattr(model.t3, method_name):
        method = getattr(model.t3, method_name)
        try:
            result = method([text])
            print(f"  model.t3.{method_name}([text]): {result}")
        except Exception as e:
            print(f"  model.t3.{method_name}([text]): Error - {e}")

# Generate and check duration
print("\n\nGenerating...")
wav = model.generate(text, audio_prompt_path=ref_path)
print(f"Duration: {wav.shape[-1] / 24000:.2f}s")
