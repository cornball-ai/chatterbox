#!/usr/bin/env python3
"""Explore encoder forward method to understand data flow."""

import torch
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
encoder = model.s3gen.flow.encoder

# Get the encoder source code
import chatterbox.models.s3gen.transformer.upsample_encoder as upsample_module
print("\n=== UpsampleConformerEncoder.forward source ===")
print(inspect.getsource(encoder.forward))

print("\n=== Upsample1D.forward source ===")
print(inspect.getsource(encoder.up_layer.forward))

print("\n=== PreLookaheadLayer.forward source ===")
print(inspect.getsource(encoder.pre_lookahead_layer.forward))
