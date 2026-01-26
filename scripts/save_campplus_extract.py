#!/usr/bin/env python3
"""Trace CAMPPlus extract_feature function."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen
speaker_encoder = s3gen.speaker_encoder

# Find extract_feature
print("\n=== Finding extract_feature ===")
# Look in the module where CAMPPlus is defined
campplus_module = type(speaker_encoder).__module__
print(f"CAMPPlus module: {campplus_module}")

# Import and inspect
import importlib
mod = importlib.import_module(campplus_module)
print(f"\nModule contents with 'extract':")
for name in dir(mod):
    if 'extract' in name.lower():
        print(f"  {name}")

# Get extract_feature source
if hasattr(mod, 'extract_feature'):
    extract_feature = mod.extract_feature
    print(f"\nextract_feature source:")
    try:
        source = inspect.getsource(extract_feature)
        print(source)
    except Exception as e:
        print(f"Error: {e}")

# Check for FBank or similar
print("\n=== Looking for FBank/Mel computation ===")
for name in dir(mod):
    if 'bank' in name.lower() or 'fbank' in name.lower() or 'mel' in name.lower():
        print(f"  {name}")
        obj = getattr(mod, name, None)
        if obj and hasattr(obj, '__call__'):
            try:
                sig = inspect.signature(obj)
                print(f"    signature: {sig}")
            except:
                pass

# Load reference audio and trace
print("\n=== Tracing extract_feature ===")
ref = load_file("/outputs/mel_reference.safetensors")
wav_np = ref["audio_wav"].numpy()
ref_wav_tensor = torch.from_numpy(wav_np).float().cuda()
print(f"ref_wav shape: {ref_wav_tensor.shape}")

with torch.inference_mode():
    # Call extract_feature - it expects a list of audio tensors
    print("\nCalling extract_feature:")
    audio_list = [ref_wav_tensor]  # List of 1D tensors
    speech, speech_lengths, speech_times = mod.extract_feature(audio_list)
    print(f"  speech shape: {speech.shape}")
    print(f"  speech_lengths: {speech_lengths}")
    print(f"  speech_times: {speech_times}")
    print(f"  speech mean: {speech.mean().item():.6f}, std: {speech.std().item():.6f}")

    # Save the extracted features
    save_dict = {
        "ref_wav": ref_wav_tensor.cpu().contiguous(),
        "speech_feat": speech.cpu().contiguous(),
        "speech_lengths": torch.tensor(speech_lengths).contiguous(),
    }

    # Now run through CAMPPlus forward step by step
    print("\n=== CAMPPlus Forward Step by Step ===")
    x = speech.to(torch.float32)
    print(f"Input x shape: {x.shape}")

    # Step 1: permute (B,T,F) => (B,F,T)
    x = x.permute(0, 2, 1)
    print(f"After permute: {x.shape}")
    save_dict["after_permute"] = x.cpu().contiguous()

    # Step 2: FCM head
    fcm = speaker_encoder.head
    # FCM forward: unsqueeze, conv1+bn1+relu, layer1, layer2, conv2+bn2+relu, reshape
    x_unsqueeze = x.unsqueeze(1)
    print(f"\nFCM input (unsqueeze): {x_unsqueeze.shape}")
    save_dict["fcm_input"] = x_unsqueeze.cpu().contiguous()

    x_conv1 = fcm.conv1(x_unsqueeze)
    x_bn1 = fcm.bn1(x_conv1)
    x_relu1 = torch.nn.functional.relu(x_bn1)
    print(f"After conv1+bn1+relu: {x_relu1.shape}")
    save_dict["fcm_after_conv1"] = x_relu1.cpu().contiguous()

    x_layer1 = fcm.layer1(x_relu1)
    print(f"After layer1: {x_layer1.shape}")
    save_dict["fcm_after_layer1"] = x_layer1.cpu().contiguous()

    x_layer2 = fcm.layer2(x_layer1)
    print(f"After layer2: {x_layer2.shape}")
    save_dict["fcm_after_layer2"] = x_layer2.cpu().contiguous()

    x_conv2 = fcm.conv2(x_layer2)
    x_bn2 = fcm.bn2(x_conv2)
    x_relu2 = torch.nn.functional.relu(x_bn2)
    print(f"After conv2+bn2+relu: {x_relu2.shape}")
    save_dict["fcm_after_conv2"] = x_relu2.cpu().contiguous()

    shape = x_relu2.shape
    fcm_out = x_relu2.reshape(shape[0], shape[1] * shape[2], shape[3])
    print(f"FCM output (reshape): {fcm_out.shape}")
    save_dict["fcm_output"] = fcm_out.cpu().contiguous()

    # Step 3: xvector
    xvec = speaker_encoder.xvector
    print(f"\nxvector input: {fcm_out.shape}")

    x = xvec.tdnn(fcm_out)
    print(f"After tdnn: {x.shape}")
    save_dict["xvec_after_tdnn"] = x.cpu().contiguous()

    x = xvec.block1(x)
    x = xvec.transit1(x)
    print(f"After block1+transit1: {x.shape}")

    x = xvec.block2(x)
    x = xvec.transit2(x)
    print(f"After block2+transit2: {x.shape}")

    x = xvec.block3(x)
    x = xvec.transit3(x)
    print(f"After block3+transit3: {x.shape}")
    save_dict["xvec_after_blocks"] = x.cpu().contiguous()

    x = xvec.out_nonlinear(x)
    print(f"After out_nonlinear: {x.shape}")
    save_dict["xvec_after_nonlinear"] = x.cpu().contiguous()

    x = xvec.stats(x)
    print(f"After stats: {x.shape}")
    save_dict["xvec_after_stats"] = x.cpu().contiguous()

    x = xvec.dense(x)
    print(f"After dense (final): {x.shape}")
    save_dict["embedding"] = x.cpu().contiguous()

    print(f"\nFinal embedding: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

save_file(save_dict, "/outputs/campplus_steps.safetensors")
print("\nSaved to /outputs/campplus_steps.safetensors")
print(f"Keys: {list(save_dict.keys())}")
