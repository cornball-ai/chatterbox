#!/usr/bin/env python3
"""Check how encoder creates masks."""

import torch
import inspect
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained('cuda')
encoder = tts.s3gen.flow.encoder

# Get forward source
print("Encoder forward (first 60 lines):")
source = inspect.getsource(encoder.forward)
for i, line in enumerate(source.split('\n')[:60]):
    print(f"{i+1:3d}: {line}")
