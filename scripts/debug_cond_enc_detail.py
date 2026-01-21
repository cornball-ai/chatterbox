#!/usr/bin/env python3
"""Debug T3 conditioning encoder in detail."""

import torch
from chatterbox.tts import ChatterboxTTS
import safetensors.torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    t3 = model.t3

    # Reference audio
    ref_audio = "/ref_audio/jfk.wav"

    # Prepare conditionals
    print("\nPreparing conditionals...")
    model.prepare_conditionals(ref_audio, exaggeration=0.5)
    t3_cond = model.conds.t3

    # Print t3_cond contents
    print("\nt3_cond contents:")
    print(f"  speaker_emb: {t3_cond.speaker_emb.shape if t3_cond.speaker_emb is not None else None}")
    print(f"  clap_emb: {t3_cond.clap_emb.shape if t3_cond.clap_emb is not None else None}")
    print(f"  cond_prompt_speech_tokens: {t3_cond.cond_prompt_speech_tokens.shape if t3_cond.cond_prompt_speech_tokens is not None else None}")
    print(f"  cond_prompt_speech_emb: {t3_cond.cond_prompt_speech_emb.shape if t3_cond.cond_prompt_speech_emb is not None else None}")
    print(f"  emotion_adv: {t3_cond.emotion_adv}")

    # Simply call prepare_input_embeds to see what cond_enc produces
    print("\n=== Using t3.prepare_input_embeds ===")

    import torch.nn.functional as F

    # Text tokens (same as in llama_layers.py)
    text = "Hello world"
    text_tokens = model.tokenizer.text_to_tokens(text).to(device)
    eot = t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)

    # BOS token
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_cfg = torch.cat([bos, bos], dim=0)

    # Prepare embeddings - this calls cond_enc internally
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens_cfg,
        speech_tokens=bos_cfg,
        cfg_weight=0.5,
    )

    print(f"Input embeds shape: {embeds.shape}")
    print(f"Conditioning length: {len_cond}")

    # Extract just the conditioning part
    cond_emb = embeds[0, :len_cond, :]
    print(f"Conditioning embedding: {cond_emb.shape}")
    print(f"  mean: {cond_emb.mean().item():.6f}")
    print(f"  std: {cond_emb.std().item():.6f}")

    result = embeds[:, :len_cond, :]
    cond_spkr = embeds[:, :1, :]  # First position should be speaker
    prompt_emb = embeds[:, 1:len_cond-1, :]  # Middle should be perceiver output
    cond_emotion = embeds[:, len_cond-1:len_cond, :]  # Last should be emotion

    print(f"\nBreakdown (estimated):")
    print(f"  Speaker (pos 0): mean={cond_spkr[0].mean().item():.6f}, std={cond_spkr[0].std().item():.6f}")
    print(f"  Perceiver (pos 1-{len_cond-2}): mean={prompt_emb[0].mean().item():.6f}, std={prompt_emb[0].std().item():.6f}")
    print(f"  Emotion (pos {len_cond-1}): mean={cond_emotion[0].mean().item():.6f}, std={cond_emotion[0].std().item():.6f}")

    # Save for R comparison
    outputs = {
        "speaker_emb": t3_cond.speaker_emb.cpu().float(),
        "cond_prompt_speech_tokens": t3_cond.cond_prompt_speech_tokens.cpu() if t3_cond.cond_prompt_speech_tokens is not None else torch.zeros(1),
        "cond_full": result.cpu().float(),
        "input_embeds": embeds.cpu().float(),
        "len_cond": torch.tensor([len_cond]),
    }

    safetensors.torch.save_file(outputs, "/outputs/cond_enc_debug.safetensors")
    print("\nSaved to /outputs/cond_enc_debug.safetensors")

if __name__ == "__main__":
    main()
