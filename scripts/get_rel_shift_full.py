#!/usr/bin/env python3
"""Get full rel_shift source."""

import inspect
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained('cuda')
attn = tts.s3gen.flow.encoder.encoders[0].self_attn

print("Full rel_shift source:")
print(inspect.getsource(attn.rel_shift))
