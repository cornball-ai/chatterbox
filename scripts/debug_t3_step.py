#!/usr/bin/env python3
"""Debug T3 inference step-by-step, saving intermediate values."""

import torch
import safetensors.torch
import numpy as np
import sys
import os

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond
import torch.nn.functional as F
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, MinPLogitsWarper

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading Chatterbox model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    t3 = model.t3

    # Reference audio path
    ref_audio = "/ref_audio/ShortCasey.wav"

    # Test text
    text = "Hello, this is a test of the text to speech system."
    text = punc_norm(text)

    # Prepare conditionals
    print("Preparing conditionals...")
    model.prepare_conditionals(ref_audio, exaggeration=0.5)
    t3_cond = model.conds.t3

    # Tokenize text
    text_tokens = model.tokenizer.text_to_tokens(text).to(device)
    sot = t3.hp.start_text_token
    eot = t3.hp.stop_text_token
    text_tokens_padded = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens_padded = F.pad(text_tokens_padded, (0, 1), value=eot)

    # Double for CFG
    cfg_weight = 0.5
    text_tokens_cfg = torch.cat([text_tokens_padded, text_tokens_padded], dim=0)

    print(f"Text tokens shape: {text_tokens_cfg.shape}")

    # Set seed
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # Prepare initial embeddings (same as in inference)
    bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)

    # Prepare embeddings
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens_cfg,
        speech_tokens=bos_token.expand(2, -1),
        cfg_weight=cfg_weight,
    )
    print(f"Embeddings shape: {embeds.shape}")
    print(f"Conditioning length: {len_cond}")

    # BOS embedding for CFG
    bos_embed = t3.speech_emb(bos_token)
    bos_embed = bos_embed + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])

    # Combine condition and BOS
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
    print(f"Initial inputs_embeds shape: {inputs_embeds.shape}")

    # Create patched model
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=None,
    )

    # Create logits processors
    top_p_warper = TopPLogitsWarper(top_p=0.9)
    min_p_warper = MinPLogitsWarper(min_p=0.05)
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=1.2)

    # Initial forward pass
    with torch.inference_mode():
        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        past = output.past_key_values
        generated_ids = bos_token.clone()

        # Get logits from first step
        logits_raw = output.logits[:, -1, :]
        print(f"Raw logits shape: {logits_raw.shape}")
        print(f"Raw logits mean: {logits_raw.mean().item():.6f}")
        print(f"Raw logits std: {logits_raw.std().item():.6f}")
        print(f"Raw logits range: [{logits_raw.min().item():.4f}, {logits_raw.max().item():.4f}]")

        # CFG combine
        cond = logits_raw[0:1, :]
        uncond = logits_raw[1:2, :]
        cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
        logits_cfg = cond + cfg * (cond - uncond)
        print(f"\nAfter CFG:")
        print(f"Logits shape: {logits_cfg.shape}")
        print(f"Logits mean: {logits_cfg.mean().item():.6f}")
        print(f"Logits range: [{logits_cfg.min().item():.4f}, {logits_cfg.max().item():.4f}]")

        # Apply repetition penalty
        ids_for_proc = generated_ids[:1, ...]
        logits_rep = repetition_penalty_processor(ids_for_proc, logits_cfg.clone())
        print(f"\nAfter repetition penalty:")
        print(f"Logits range: [{logits_rep.min().item():.4f}, {logits_rep.max().item():.4f}]")

        # Apply temperature
        temperature = 0.8
        logits_temp = logits_rep / temperature
        print(f"\nAfter temperature ({temperature}):")
        print(f"Logits range: [{logits_temp.min().item():.4f}, {logits_temp.max().item():.4f}]")

        # Apply min_p and top_p
        logits_minp = min_p_warper(ids_for_proc, logits_temp.clone())
        logits_topp = top_p_warper(ids_for_proc, logits_minp.clone())
        print(f"\nAfter min_p and top_p filtering:")
        print(f"Non-inf count: {(logits_topp > -float('inf')).sum().item()}")

        # Convert to probs
        probs = torch.softmax(logits_topp, dim=-1)
        print(f"\nProbabilities:")
        print(f"Max prob: {probs.max().item():.6f}")
        print(f"Min prob (non-zero): {probs[probs > 0].min().item():.6f}")
        print(f"Non-zero count: {(probs > 0).sum().item()}")

        # Get top 10 tokens by probability
        top_probs, top_indices = probs.topk(10)
        print(f"\nTop 10 tokens by probability:")
        for i in range(10):
            print(f"  Token {top_indices[0, i].item()}: {top_probs[0, i].item():.6f}")

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        print(f"\nSampled token: {next_token.item()}")

        # Save debug info for 5 more steps
        all_sampled = [next_token.item()]
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        for step in range(1, 6):
            # Get next embedding
            next_token_embed = t3.speech_emb(next_token)
            next_token_embed = next_token_embed + t3.speech_pos_emb.get_fixed_embedding(step)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward
            output = patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

            # Process logits (same as first step)
            logits_step = output.logits[:, -1, :]
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            logits = cond + cfg * (cond - uncond)

            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)
            logits = logits / temperature
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            all_sampled.append(next_token.item())
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == t3.hp.stop_speech_token:
                print(f"\nEOS detected at step {step}")
                break

        print(f"\nFirst 6 sampled tokens: {all_sampled}")

    # Save intermediate values
    outputs = {
        "raw_logits_step0": logits_raw.cpu().float(),
        "logits_after_cfg": logits_cfg.cpu().float(),
        "logits_after_temp": logits_temp.cpu().float(),
        "probs_step0": probs.cpu().float(),
        "sampled_tokens": torch.tensor(all_sampled),
        "speaker_emb": t3_cond.speaker_emb.cpu().float(),
        "initial_embeds": embeds.cpu().float(),
    }

    output_path = "/outputs/t3_debug.safetensors"
    safetensors.torch.save_file(outputs, output_path)
    print(f"\nSaved debug outputs to {output_path}")

if __name__ == "__main__":
    main()
