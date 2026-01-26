#!/usr/bin/env python3
"""Check estimator forward up block section."""

import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get forward source
src = inspect.getsource(type(estimator).forward)

# Find and print the up block section
lines = src.split('\n')
in_up_block = False
for i, line in enumerate(lines):
    if 'up_block' in line.lower() or in_up_block:
        print(f"{i+1:3}: {line}")
        if 'for' in line and 'up_block' in line:
            in_up_block = True
        if in_up_block and line.strip().startswith('return'):
            break
        if in_up_block and i > 65 and 'for' in line and 'up_block' not in line:
            in_up_block = False
