#!/usr/bin/env python3
"""Print full estimator forward source."""

import inspect
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

src = inspect.getsource(type(estimator).forward)
lines = src.split('\n')

# Print lines 60-90 (up blocks + final)
for i in range(60, min(95, len(lines))):
    print(f"{i+1:3}: {lines[i]}")
