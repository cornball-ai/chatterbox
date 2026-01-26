#!/usr/bin/env python3
"""Check GELU implementation in Python model."""

import torch
import torch.nn.functional as F
from chatterbox.tts import ChatterboxTTS

print("Loading model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

# Get GELU from FF
gelu = estimator.down_blocks[0][1][0].ff.net[0]
print(f"GELU type: {type(gelu).__name__}")
print(f"GELU attributes: {[a for a in dir(gelu) if not a.startswith('_')][:20]}")

# Test GELU behavior
x = torch.tensor([[0.0, 0.5, 1.0, -0.5, -1.0]], device='cuda')

# Check if it's GELU or GEGLU
if hasattr(gelu, 'proj'):
    print(f"Has proj: {gelu.proj}")
else:
    print("No proj attribute")

# Test with standard GELU
with torch.no_grad():
    gelu_none = F.gelu(x, approximate='none')
    gelu_tanh = F.gelu(x, approximate='tanh')
    print(f"GELU none: {gelu_none}")
    print(f"GELU tanh: {gelu_tanh}")

# Check what the model uses
import inspect
try:
    src = inspect.getsource(type(gelu).forward)
    print(f"\nGELU forward source:\n{src}")
except:
    print("Could not get source")

# Test actual forward
with torch.no_grad():
    test_input = torch.randn(1, 50, 256, device='cuda')
    proj_out = gelu.proj(test_input)
    print(f"\nAfter proj: shape={proj_out.shape}, mean={proj_out.mean().item():.4f}")

    # Apply GELU (the net[0] is GELU not GEGLU based on architecture trace)
    gelu_out = gelu(test_input)
    print(f"After GELU: shape={gelu_out.shape}, mean={gelu_out.mean().item():.4f}")
