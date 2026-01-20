#!/usr/bin/env python3
"""Extract T3 reference tokens for comparison with R implementation."""

import torch
import safetensors.torch
import numpy as np
import sys
import os

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond
import torch.nn.functional as F

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading Chatterbox model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Reference audio path (mounted from host)
    ref_audio = "/ref_audio/ShortCasey.wav"
    if not os.path.exists(ref_audio):
        print(f"Reference audio not found: {ref_audio}")
        sys.exit(1)

    # Test text
    text = "Hello, this is a test of the text to speech system."
    print(f"Original text: {text}")
    text = punc_norm(text)
    print(f"Normalized text: {text}")

    # Prepare conditionals (voice embedding + S3Gen ref)
    print("Preparing conditionals...")
    model.prepare_conditionals(ref_audio, exaggeration=0.5)

    # Get raw inputs for comparison
    t3_cond = model.conds.t3
    print(f"Speaker embedding shape: {t3_cond.speaker_emb.shape}")
    print(f"Speaker embedding mean: {t3_cond.speaker_emb.mean().item():.6f}")
    print(f"Speaker embedding std: {t3_cond.speaker_emb.std().item():.6f}")

    if t3_cond.cond_prompt_speech_tokens is not None:
        print(f"Cond prompt tokens shape: {t3_cond.cond_prompt_speech_tokens.shape}")
        print(f"Cond prompt tokens first 20: {t3_cond.cond_prompt_speech_tokens[0, :20].tolist()}")

    # Tokenize text (same as generate() does)
    text_tokens = model.tokenizer.text_to_tokens(text).to(device)
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Text tokens: {text_tokens[0].tolist()}")

    # Add start/stop tokens (as generate() does)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    print(f"Start text token: {sot}, Stop text token: {eot}")

    text_tokens_padded = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens_padded = F.pad(text_tokens_padded, (0, 1), value=eot)
    print(f"Text tokens with SOT/EOT: {text_tokens_padded[0].tolist()}")

    # For CFG, double the batch
    cfg_weight = 0.5
    if cfg_weight > 0.0:
        text_tokens_cfg = torch.cat([text_tokens_padded, text_tokens_padded], dim=0)
    else:
        text_tokens_cfg = text_tokens_padded

    # Generate speech tokens with fixed seed
    print("\nGenerating speech tokens...")
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens_cfg,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=cfg_weight,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=0.9,
        )
        # Extract conditional batch
        speech_tokens = speech_tokens[0]

    print(f"\nRaw speech tokens shape: {speech_tokens.shape}")
    print(f"Speech tokens (first 50): {speech_tokens[:50].tolist()}")
    print(f"Speech tokens (last 50): {speech_tokens[-50:].tolist()}")
    print(f"Speech tokens min: {speech_tokens.min().item()}")
    print(f"Speech tokens max: {speech_tokens.max().item()}")
    print(f"Total tokens: {len(speech_tokens)}")

    # Token statistics
    unique_tokens = torch.unique(speech_tokens)
    print(f"Unique tokens: {len(unique_tokens)}")
    print(f"Token histogram (top 10):")
    values, counts = torch.unique(speech_tokens, return_counts=True)
    sorted_idx = counts.argsort(descending=True)[:10]
    for idx in sorted_idx:
        print(f"  Token {values[idx].item()}: {counts[idx].item()} times")

    # Save outputs for comparison
    outputs = {
        "speaker_emb": t3_cond.speaker_emb.cpu().float(),
        "text_tokens": text_tokens[0].cpu(),
        "text_tokens_with_sot_eot": text_tokens_padded[0].cpu(),
        "speech_tokens": speech_tokens.cpu(),
    }

    if t3_cond.cond_prompt_speech_tokens is not None:
        outputs["cond_prompt_speech_tokens"] = t3_cond.cond_prompt_speech_tokens.cpu()

    output_path = "/outputs/t3_reference.safetensors"
    safetensors.torch.save_file(outputs, output_path)
    print(f"\nSaved reference to {output_path}")

if __name__ == "__main__":
    main()
