#!/usr/bin/env python3
"""Save S3Gen component steps for R comparison."""

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
# Step 1: embed_ref method
# ============================================================================
print("\n=== Step 1: embed_ref Method ===")
try:
    embed_ref_source = inspect.getsource(s3gen.embed_ref)
    print("embed_ref source:")
    print(embed_ref_source)
except Exception as e:
    print(f"Error getting embed_ref source: {e}")

# ============================================================================
# Step 2: Call embed_ref and inspect outputs
# ============================================================================
print("\n=== Step 2: embed_ref Outputs ===")
with torch.inference_mode():
    ref_dict = s3gen.embed_ref(ref_wav_tensor, 16000)
    print(f"ref_dict keys: {ref_dict.keys()}")
    for k, v in ref_dict.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

# ============================================================================
# Step 3: flow.inference signature and source
# ============================================================================
print("\n=== Step 3: flow.inference ===")
flow = s3gen.flow
try:
    sig = inspect.signature(flow.inference)
    print(f"flow.inference signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

try:
    flow_inf_source = inspect.getsource(flow.inference)
    print("\nflow.inference source:")
    print(flow_inf_source)
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Step 4: Trace through flow.inference step by step
# ============================================================================
print("\n=== Step 4: Flow Inference Trace ===")
with torch.inference_mode():
    speech_tokens = test_tokens
    token_len = torch.LongTensor([speech_tokens.size(1)]).cuda()

    # Step 4a: Token embedding
    print("\n4a. Token embedding:")
    token_emb = flow.input_embedding(speech_tokens)
    print(f"  token_emb shape: {token_emb.shape}")

    # Step 4b: Speaker embedding projection
    print("\n4b. Speaker embedding projection:")
    embedding = ref_dict['embedding']
    embedding_norm = torch.nn.functional.normalize(embedding, dim=1)
    spk_emb = flow.spk_embed_affine_layer(embedding_norm)
    print(f"  embedding shape: {embedding.shape}")
    print(f"  embedding_norm shape: {embedding_norm.shape}")
    print(f"  spk_emb shape: {spk_emb.shape}")

    # Step 4c: Check what encoder expects
    print("\n4c. Encoder (UpsampleConformerEncoder):")
    try:
        enc_source = inspect.getsource(type(flow.encoder).forward)
        lines = enc_source.split('\n')[:40]
        print("encoder.forward source (first 40 lines):")
        for line in lines:
            print(f"  {line}")
    except Exception as e:
        print(f"Error: {e}")

    # Step 4d: Check decoder (CausalConditionalCFM)
    print("\n4d. Decoder (CausalConditionalCFM):")
    try:
        dec_source = inspect.getsource(type(flow.decoder).inference)
        print("decoder.inference source:")
        print(dec_source)
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# Step 5: Run full inference and save intermediates
# ============================================================================
print("\n=== Step 5: Full Inference with Intermediates ===")

# We need to hook into the flow to capture intermediates
# Let's trace through manually

with torch.inference_mode():
    speech_tokens = test_tokens
    token_len = torch.LongTensor([speech_tokens.size(1)]).cuda()

    # Call flow.inference to get output
    output_mels, output_lens = flow.inference(
        token=speech_tokens,
        token_len=token_len,
        finalize=False,
        **ref_dict,
    )
    print(f"output_mels shape: {output_mels.shape}")
    print(f"output_lens: {output_lens}")

    # Now run HiFTGenerator
    print("\n5b. HiFTGenerator:")
    hift = s3gen.mel2wav
    cache_source = torch.zeros(1, 1, 0).cuda()
    output_wav, s = hift.inference(speech_feat=output_mels, cache_source=cache_source)
    print(f"output_wav shape: {output_wav.shape}")

# ============================================================================
# Step 6: Save all intermediates
# ============================================================================
print("\n=== Step 6: Save Intermediates ===")
save_dict = {
    "ref_wav": ref_wav_tensor.cpu().contiguous(),
    "test_tokens": test_tokens.cpu().contiguous(),
    "embedding": ref_dict['embedding'].cpu().contiguous(),
    "embedding_norm": embedding_norm.cpu().contiguous(),
    "prompt_token": ref_dict['prompt_token'].cpu().contiguous(),
    "prompt_token_len": ref_dict['prompt_token_len'].cpu().contiguous(),
    "prompt_feat": ref_dict['prompt_feat'].cpu().contiguous(),
    "output_mels": output_mels.cpu().contiguous(),
    "output_wav": output_wav.cpu().contiguous(),
    "token_emb": token_emb.cpu().contiguous(),
    "spk_emb": spk_emb.cpu().contiguous(),
}

save_file(save_dict, "/outputs/s3gen_components.safetensors")
print("Saved to /outputs/s3gen_components.safetensors")
print(f"\nSaved keys: {list(save_dict.keys())}")
