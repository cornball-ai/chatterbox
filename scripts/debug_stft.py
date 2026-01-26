#!/usr/bin/env python3
"""Debug STFT parameters in Python chatterbox."""

import inspect
from chatterbox.models.voice_encoder import melspec

print("=== _stft source ===")
print(inspect.getsource(melspec._stft))

print("\n=== melspectrogram source ===")
print(inspect.getsource(melspec.melspectrogram))

# Also check if there's any preemphasis
print("\n=== preemphasis source (if exists) ===")
if hasattr(melspec, 'preemphasis'):
    print(inspect.getsource(melspec.preemphasis))
else:
    print("No preemphasis function")
