#!/usr/bin/env python3
"""Check forward_layers method."""

import torch
import inspect
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained('cuda')
encoder = tts.s3gen.flow.encoder

print("forward_layers source:")
source = inspect.getsource(encoder.forward_layers)
for i, line in enumerate(source.split('\n')[:30]):
    print(f"{i+1:3d}: {line}")

print("\nadd_optional_chunk_mask:")
from chatterbox.models.s3gen.transformer.subsampling import add_optional_chunk_mask
source2 = inspect.getsource(add_optional_chunk_mask)
for i, line in enumerate(source2.split('\n')[:40]):
    print(f"{i+1:3d}: {line}")
