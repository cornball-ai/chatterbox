#!/usr/bin/env python3
"""Check Python T3 inference default parameters"""

import sys
sys.path.insert(0, '/app/.venv/lib/python3.11/site-packages')

from chatterbox.tts import ChatterboxTTS
import inspect

# Get T3 inference signature
model = ChatterboxTTS.from_pretrained("cuda")

# Check ChatterboxTTS.generate
print("=== ChatterboxTTS.generate ===")
sig = inspect.signature(model.generate)
for name, param in sig.parameters.items():
    if param.default is not inspect.Parameter.empty:
        print(f"  {name}: {param.default}")

# List T3 methods
print("\n=== T3 methods ===")
for attr in dir(model.t3):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Check T3.inference if it exists
if hasattr(model.t3, 'inference'):
    print("\n=== T3.inference signature ===")
    sig = inspect.signature(model.t3.inference)
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            print(f"  {name}: {param.default}")

# Check what happens during inference
print("\n=== Inference trace ===")
text = "Hello world"
audio_prompt = "/app/.venv/lib/python3.11/site-packages/chatterbox/assets/sam_altman_3.wav"

# Load reference
ref = model.load_reference(audio_prompt)
print(f"Reference ve_embedding shape: {ref['ve_embedding'].shape}")
print(f"Reference cond_prompt_speech_tokens shape: {ref['cond_prompt_speech_tokens'].shape}")

# Prepare text
text_tokens = model.t3.tokenize_text([text])
print(f"Text tokens shape: {text_tokens.shape}")
print(f"Text token ids: {text_tokens.tolist()}")

# Cond
cond = model.t3.prepare_cond(**ref)
print(f"\nCond keys: {list(cond.keys())}")

# Generate with timing
import torch
torch.cuda.synchronize()
import time
start = time.time()

# Generate speech - use the model's generate method
wav = model.generate(text, audio_prompt_path=audio_prompt)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"\nGenerated audio in {elapsed:.2f}s")
print(f"Audio shape: {wav.shape}")
print(f"Audio duration: {wav.shape[-1] / 24000:.2f}s")
