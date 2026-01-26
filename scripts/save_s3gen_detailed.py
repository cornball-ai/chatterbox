#!/usr/bin/env python3
"""Save S3Gen detailed computation steps for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import scipy.io.wavfile as wavfile
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen

# Load reference audio
print("\nLoading reference audio...")
ref = load_file("/outputs/mel_reference.safetensors")
wav_np = ref["audio_wav"].numpy()
print(f"Reference audio: {len(wav_np)} samples")

# Create tensors
ref_wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).float().cuda()
torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (1, 31), device='cuda')
print(f"Test tokens: {test_tokens.shape}")

# ============================================================================
# Step 1: Class hierarchy
# ============================================================================
print("\n=== Step 1: Class Hierarchy ===")
print(f"S3Gen class: {type(s3gen).__name__}")
print(f"S3Gen bases: {type(s3gen).__bases__}")
print(f"S3Gen MRO:")
for cls in type(s3gen).__mro__:
    print(f"  {cls.__name__}")

# ============================================================================
# Step 2: Parent class forward (S3Token2Wav probably inherits from flow logic)
# ============================================================================
print("\n=== Step 2: Parent Class Forward ===")
# Get the parent's forward method
parent_class = type(s3gen).__bases__[0]
print(f"Parent class: {parent_class.__name__}")

try:
    parent_source = inspect.getsource(parent_class.forward)
    lines = parent_source.split('\n')[:100]
    print(f"\n{parent_class.__name__}.forward source (first 100 lines):")
    for line in lines:
        print(f"  {line}")
except Exception as e:
    print(f"Could not get parent source: {e}")

# ============================================================================
# Step 3: CAMPPlus details
# ============================================================================
print("\n=== Step 3: CAMPPlus Details ===")
se = s3gen.speaker_encoder
print(f"CAMPPlus type: {type(se).__name__}")
print(f"Input expected: mel spectrogram (B, T, F)")
print(f"Output: speaker embedding")

# Check head and xvector
print(f"\nhead (FCM):")
for name, child in se.head.named_children():
    print(f"  {name}: {type(child).__name__}")
print(f"\nxvector:")
for name, child in se.xvector.named_children():
    print(f"  {name}: {type(child).__name__}")

# ============================================================================
# Step 4: Flow module (CausalMaskedDiffWithXvec)
# ============================================================================
print("\n=== Step 4: CausalMaskedDiffWithXvec Details ===")
flow = s3gen.flow
print(f"Flow type: {type(flow).__name__}")
print(f"\nFlow children:")
for name, child in flow.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in list(child.named_children())[:5]:
            print(f"    {n2}: {type(c2).__name__}")

# Check flow forward
try:
    flow_source = inspect.getsource(type(flow).forward)
    lines = flow_source.split('\n')[:50]
    print(f"\nflow.forward source (first 50 lines):")
    for line in lines:
        print(f"  {line}")
except Exception as e:
    print(f"Could not get flow source: {e}")

# Check flow inference/sample method
print("\n\nLooking for inference/sample methods in flow:")
for method in ['forward', 'inference', 'sample', 'generate', '__call__']:
    if hasattr(flow, method):
        print(f"  has {method}")

# ============================================================================
# Step 5: HiFTGenerator details
# ============================================================================
print("\n=== Step 5: HiFTGenerator Details ===")
hift = s3gen.mel2wav
print(f"HiFTGenerator type: {type(hift).__name__}")
print(f"\nHiFTGenerator children:")
for name, child in hift.named_children():
    print(f"  {name}: {type(child).__name__}")

# Check inference method
try:
    hift_source = inspect.getsource(type(hift).inference)
    lines = hift_source.split('\n')[:60]
    print(f"\nhift.inference source (first 60 lines):")
    for line in lines:
        print(f"  {line}")
except Exception as e:
    print(f"Could not get hift inference source: {e}")

# ============================================================================
# Step 6: Trace S3Gen.forward with actual inputs
# ============================================================================
print("\n=== Step 6: Trace S3Gen Forward ===")

with torch.inference_mode():
    # Run s3gen forward and capture intermediates
    speech_tokens = test_tokens
    ref_wav = ref_wav_tensor
    ref_sr = 16000

    # Get speaker embedding from ref wav
    # We need to compute mel first for CAMPPlus
    from chatterbox.models.s3tokenizer import S3Tokenizer

    # Check how ref embedding is computed in parent forward
    print("\nTracing computation...")

    # The parent class should have the logic
    # Let's manually trace by looking at what happens in super().forward()

    # Looking at the parent source, it should:
    # 1. Compute mel from speech_tokens
    # 2. Compute ref_s from ref_wav
    # 3. Run CFM flow

    # Let's try to call the components step by step

    # Step A: tokenizer.quantizer.decode (speech_tokens -> mel?)
    print("\nA. Checking tokenizer.quantizer:")
    tokenizer = s3gen.tokenizer
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    if hasattr(tokenizer, 'quantizer'):
        print(f"  quantizer type: {type(tokenizer.quantizer).__name__}")
        if hasattr(tokenizer.quantizer, 'decode'):
            print(f"  quantizer.decode exists")

    # Step B: Check parent forward signature
    print("\nB. Full parent forward source:")
    try:
        parent_class = type(s3gen).__bases__[0]
        full_source = inspect.getsource(parent_class.forward)
        print(full_source)
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# Step 7: Run full s3gen and save outputs
# ============================================================================
print("\n=== Step 7: Run S3Gen and Save ===")

with torch.inference_mode():
    output = s3gen.forward(
        speech_tokens=test_tokens,
        ref_wav=ref_wav_tensor,
        ref_sr=16000,
    )
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")

# Save
save_dict = {
    "ref_wav": ref_wav_tensor.cpu().contiguous(),
    "test_tokens": test_tokens.cpu().contiguous(),
    "output_wav": output.cpu().contiguous(),
}
save_file(save_dict, "/outputs/s3gen_detailed.safetensors")
print("Saved to /outputs/s3gen_detailed.safetensors")
