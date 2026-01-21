#!/usr/bin/env python3
"""Save T3 computation step by step for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")

# Use the same audio from mel_reference and a simple text
text = "Hello world"
print(f"\nText: '{text}'")

# Load reference audio
ref = load_file("/outputs/mel_reference.safetensors")
wav_np = ref["audio_wav"].numpy()
print(f"Audio: {len(wav_np)} samples at 16kHz")

# ============================================================================
# Step 1: Voice encoder → speaker embedding
# ============================================================================
print("\n=== Step 1: Voice Encoder ===")
with torch.no_grad():
    speaker_emb = model.ve.embeds_from_wavs([wav_np], sample_rate=16000)
    speaker_emb = torch.from_numpy(speaker_emb).cuda()
print(f"Speaker embedding: {speaker_emb.shape}, mean={speaker_emb.mean().item():.6f}")

# ============================================================================
# Step 2: S3 Tokenizer → prompt speech tokens
# ============================================================================
print("\n=== Step 2: S3 Tokenizer ===")
s3_tokenizer = model.s3gen.tokenizer
max_len = model.t3.hp.speech_cond_prompt_len
with torch.no_grad():
    prompt_tokens, _ = s3_tokenizer.forward([wav_np], max_len=max_len)
    prompt_tokens = torch.atleast_2d(prompt_tokens).cuda()
print(f"Prompt tokens: {prompt_tokens.shape}")
print(f"  First 10: {prompt_tokens[0, :10].cpu().numpy()}")

# ============================================================================
# Step 3: Text tokenization
# ============================================================================
print("\n=== Step 3: Text Tokenization ===")
# Use model's tokenizer
from chatterbox.tts import punc_norm
text_normalized = punc_norm(text)
print(f"Normalized text: '{text_normalized}'")

text_tokens = model.tokenizer.text_to_tokens(text_normalized).to('cuda')
print(f"Text tokens: {text_tokens.shape}, values={text_tokens.cpu().numpy()}")

# ============================================================================
# Step 4: T3 Conditioning
# ============================================================================
print("\n=== Step 4: T3 Conditioning ===")
exaggeration = 0.5

# Build T3Cond object
t3_cond = T3Cond(
    speaker_emb=speaker_emb,
    cond_prompt_speech_tokens=prompt_tokens,
    emotion_adv=exaggeration * torch.ones(1, 1, 1),
).to(device='cuda')

# Prepare conditioning embeddings (this converts tokens to embeddings)
with torch.no_grad():
    cond = model.t3.prepare_conditioning(t3_cond)
len_cond = cond.size(1)
print(f"Conditioning: {cond.shape}")
print(f"  len_cond: {len_cond}")

# ============================================================================
# Step 5: Prepare input embeddings (no CFG first)
# ============================================================================
print("\n=== Step 5: Input Embeddings ===")

with torch.no_grad():
    # Get text embeddings (token only, no position)
    text_emb_tok_only = model.t3.text_emb(text_tokens)
    print(f"Text token embeddings: {text_emb_tok_only.shape}")

    # Get text position embeddings
    text_pos_emb = model.t3.text_pos_emb(text_tokens)
    print(f"Text position embeddings: {text_pos_emb.shape}")

    # Combined text embedding (token + position)
    text_emb = text_emb_tok_only + text_pos_emb
    print(f"Text embeddings (with pos): {text_emb.shape}")

    # Get speech token embedding for start token
    start_token = torch.tensor([[model.t3.hp.start_speech_token]], device='cuda')
    speech_emb_tok_only = model.t3.speech_emb(start_token)

    # Get speech position embeddings (position 0)
    speech_pos_emb = model.t3.speech_pos_emb.get_fixed_embedding(torch.tensor([0], device='cuda'))
    speech_emb = speech_emb_tok_only + speech_pos_emb
    print(f"Speech start embedding (with pos): {speech_emb.shape}")

    # Concatenate: cond + text + speech_start
    input_embeds = torch.cat([cond, text_emb, speech_emb], dim=1)
    print(f"Input embeds (cond + text + speech_start): {input_embeds.shape}")

# ============================================================================
# Step 6: Llama backbone forward pass
# ============================================================================
print("\n=== Step 6: Llama Forward ===")

with torch.no_grad():
    # Run through Llama transformer
    llama_output = model.t3.tfmr(
        inputs_embeds=input_embeds,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = llama_output.last_hidden_state
    print(f"Hidden states: {hidden_states.shape}")
    print(f"  mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

    # Get speech logits (last position predicts first speech token)
    speech_logits = model.t3.speech_head(hidden_states[:, -1:, :])
    print(f"Speech logits (last pos): {speech_logits.shape}")
    print(f"  mean={speech_logits.mean().item():.6f}, std={speech_logits.std().item():.6f}")

# ============================================================================
# Step 7: T3 hyperparameters
# ============================================================================
print("\n=== Step 7: T3 Hyperparameters ===")
hp = model.t3.hp
print(f"  Config type: {type(hp).__name__}")
print(f"  Available attrs: {[a for a in dir(hp) if not a.startswith('_')]}")
print(f"  start_speech_token: {hp.start_speech_token}")
print(f"  stop_speech_token: {hp.stop_speech_token}")
print(f"  speech_cond_prompt_len: {hp.speech_cond_prompt_len}")
# Get dim from the model itself
print(f"  Embedding dim: {model.t3.text_emb.weight.shape[1]}")
print(f"  Text vocab size: {model.t3.text_emb.num_embeddings}")
print(f"  Speech vocab size: {model.t3.speech_emb.num_embeddings}")

# ============================================================================
# Save for R comparison
# ============================================================================
save_dict = {
    "speaker_emb": speaker_emb.cpu().contiguous(),
    "prompt_tokens": prompt_tokens.cpu().contiguous(),
    "text_tokens": text_tokens.cpu().contiguous(),
    "cond": cond.cpu().contiguous(),
    "text_emb_tok_only": text_emb_tok_only.cpu().contiguous(),
    "text_pos_emb": text_pos_emb.cpu().contiguous(),
    "text_emb": text_emb.cpu().contiguous(),
    "speech_start_emb": speech_emb.cpu().contiguous(),
    "input_embeds": input_embeds.cpu().contiguous(),
    "hidden_states": hidden_states.cpu().contiguous(),
    "speech_logits": speech_logits.cpu().contiguous(),
    "len_cond": torch.tensor([len_cond]),
    "start_speech_token": torch.tensor([model.t3.hp.start_speech_token]),
    "stop_speech_token": torch.tensor([model.t3.hp.stop_speech_token]),
}

save_file(save_dict, "/outputs/t3_steps.safetensors")
print("\nSaved to /outputs/t3_steps.safetensors")
