#!/usr/bin/env python3
"""Save HiFiGAN vocoder reference outputs for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen
hift = s3gen.mel2wav

print(f"HiFiGAN type: {type(hift).__name__}")

# ============================================================================
# Step 1: Explore HiFiGAN architecture
# ============================================================================
print("\n=== Step 1: HiFiGAN Architecture ===")
print("\nTop-level children:")
for name, child in hift.named_children():
    print(f"  {name}: {type(child).__name__}")

# Get all attributes
print("\nModule attributes:")
attrs_to_check = [
    'sample_rate', 'sampling_rate', 'hop_size', 'hop_len', 'nb_harmonics',
    'istft_n_fft', 'istft_hop_len', 'audio_limit', 'num_kernels', 'num_upsamples',
    'lrelu_slope', 'n_fft', 'hop_length', 'upsample_rates', 'upsample_kernel_sizes'
]
for attr in attrs_to_check:
    if hasattr(hift, attr):
        val = getattr(hift, attr)
        if hasattr(val, 'tolist'):
            val = val.tolist()
        print(f"  {attr}: {val}")

# Get ALL attributes
print("\nAll attributes (non-module, non-private):")
for attr in dir(hift):
    if not attr.startswith('_') and not callable(getattr(hift, attr)):
        try:
            val = getattr(hift, attr)
            if not isinstance(val, torch.nn.Module):
                print(f"  {attr}: {val}")
        except:
            pass

# ============================================================================
# Step 2: Check upsampling factor
# ============================================================================
print("\n=== Step 2: Upsampling Analysis ===")
print(f"Number of ups: {len(hift.ups)}")
for i, up in enumerate(hift.ups):
    print(f"  ups[{i}]: stride={up.stride}, kernel={up.kernel_size}, weight={up.weight.shape}")

print(f"\nNumber of source_downs: {len(hift.source_downs)}")
for i, sd in enumerate(hift.source_downs):
    print(f"  source_downs[{i}]: stride={sd.stride}, kernel={sd.kernel_size}, weight={sd.weight.shape}")

print(f"\nResblocks: {len(hift.resblocks)}")

# ============================================================================
# Step 3: Check inference source code
# ============================================================================
print("\n=== Step 3: Inference Method Source ===")
try:
    source_code = inspect.getsource(type(hift).inference)
    print(source_code[:4000])
except Exception as e:
    print(f"Could not get source: {e}")

# ============================================================================
# Step 4: Check decode method source code
# ============================================================================
print("\n=== Step 4: Decode Method Source ===")
try:
    source_code = inspect.getsource(type(hift).decode)
    print(source_code[:4000])
except Exception as e:
    print(f"Could not get source: {e}")

# ============================================================================
# Step 5: Run full inference and save outputs
# ============================================================================
print("\n=== Step 5: Run Inference and Save ===")

with torch.inference_mode():
    # Create random mel input (B, 80, T)
    torch.manual_seed(42)
    test_mel = torch.randn(1, 80, 100, device='cuda')

    print(f"Input mel shape: {test_mel.shape}")
    print(f"Input mel mean: {test_mel.mean().item():.6f}, std: {test_mel.std().item():.6f}")

    # Run inference
    result = hift.inference(test_mel)
    audio = result[0] if isinstance(result, tuple) else result
    source_cache = result[1] if isinstance(result, tuple) and len(result) > 1 else None

    print(f"Audio shape: {audio.shape}")
    print(f"Audio mean: {audio.mean().item():.6f}, std: {audio.std().item():.6f}")
    print(f"Audio min: {audio.min().item():.6f}, max: {audio.max().item():.6f}")

    if source_cache is not None:
        print(f"Source cache shape: {source_cache.shape}")

    # F0 intermediate
    f0 = hift.f0_predictor(test_mel)
    print(f"\nF0 shape: {f0.shape}")
    print(f"F0 mean: {f0.mean().item():.6f}, std: {f0.std().item():.6f}")

    # F0 upsampled
    f0_up = hift.f0_upsamp(f0.unsqueeze(1))
    print(f"F0 up shape: {f0_up.shape}")

    # Source module
    f0_for_source = f0_up.transpose(1, 2)  # (B, T, 1)
    source_result = hift.m_source(f0_for_source)
    source = source_result[0].transpose(1, 2)  # sine_merge
    print(f"Source shape: {source.shape}")
    print(f"Source mean: {source.mean().item():.6f}, std: {source.std().item():.6f}")

    # conv_pre output
    conv_pre_out = hift.conv_pre(test_mel)
    print(f"conv_pre output: {conv_pre_out.shape}")

    save_dict = {
        "input_mel": test_mel.cpu().contiguous(),
        "f0": f0.cpu().contiguous(),
        "f0_up": f0_up.cpu().contiguous(),
        "source": source.cpu().contiguous(),
        "conv_pre_out": conv_pre_out.cpu().contiguous(),
        "output_audio": audio.cpu().contiguous(),
    }

    if source_cache is not None:
        save_dict["source_cache"] = source_cache.cpu().contiguous()

    save_file(save_dict, "/outputs/hifigan_reference.safetensors")
    print("\nSaved to /outputs/hifigan_reference.safetensors")

# ============================================================================
# Step 6: Weight shapes summary
# ============================================================================
print("\n=== Step 6: Weight Shapes Summary ===")
print("\nKey weight shapes:")
print(f"  conv_pre.weight: {hift.conv_pre.weight.shape}")
print(f"  conv_post.weight: {hift.conv_post.weight.shape}")

print("\nUpsampling layers:")
for i, up in enumerate(hift.ups):
    print(f"  ups.{i}.weight: {up.weight.shape}")

print("\nSource downsampling:")
for i, sd in enumerate(hift.source_downs):
    print(f"  source_downs.{i}.weight: {sd.weight.shape}")

print("\nSource resblocks:")
for i, srb in enumerate(hift.source_resblocks):
    print(f"  source_resblocks.{i}: {type(srb).__name__}")
    if hasattr(srb, 'convs1'):
        print(f"    convs1[0].weight: {srb.convs1[0].weight.shape}")

print("\nF0 predictor:")
print(f"  f0_predictor.classifier.weight: {hift.f0_predictor.classifier.weight.shape}")

print("\nSource module:")
print(f"  m_source.l_linear.weight: {hift.m_source.l_linear.weight.shape}")
