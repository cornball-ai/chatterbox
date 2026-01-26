#!/usr/bin/env python3
"""Check encoder layer forward signature."""

import inspect
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained('cuda')
enc0 = tts.s3gen.flow.encoder.encoders[0]

print("Encoder layer class:", enc0.__class__.__name__)
print("\nForward signature:")
print(inspect.signature(enc0.forward))

print("\nForward source (first 30 lines):")
source = inspect.getsource(enc0.forward)
for i, line in enumerate(source.split('\n')[:30]):
    print(f"{i+1:3d}: {line}")
