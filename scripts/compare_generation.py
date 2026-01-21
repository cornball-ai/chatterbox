#!/usr/bin/env python3
"""Generate speech tokens and save for R comparison."""

import torch
import safetensors.torch
from chatterbox.tts import ChatterboxTTS
import torch.nn.functional as F

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    t3 = model.t3

    # Prepare conditioning
    ref_audio = "/ref_audio/jfk.wav"
    model.prepare_conditionals(ref_audio, exaggeration=0.5)
    t3_cond = model.conds.t3

    # Text
    text = "Hello world"
    text_tokens = model.tokenizer.text_to_tokens(text).to(device)
    eot = t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)

    # BOS
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_cfg = torch.cat([bos, bos], dim=0)

    # Prepare embeddings
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens_cfg,
        speech_tokens=bos_cfg,
        cfg_weight=0.5,
    )

    # Set seed for reproducibility
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # Generation parameters
    max_new_tokens = 200
    temperature = 0.8
    cfg_weight = 0.5
    top_p = 0.95
    min_p = 0.05

    print(f"\nGenerating up to {max_new_tokens} speech tokens...")

    with torch.inference_mode():
        # Initial forward
        output = t3.tfmr(
            inputs_embeds=embeds,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = output.past_key_values

        generated_ids = bos[:1].clone()
        predicted = []

        for i in range(max_new_tokens):
            # Get logits
            last_hidden = output.last_hidden_state[:, -1, :]
            logits = t3.speech_head(last_hidden.clone())

            # CFG
            cond_logits = logits[0:1, :]
            uncond_logits = logits[1:2, :]
            logits_combined = cond_logits + cfg_weight * (cond_logits - uncond_logits)

            # Temperature
            logits_temp = logits_combined / temperature

            # Softmax
            probs = torch.softmax(logits_temp, dim=-1)

            # Min-p
            max_prob = probs.max()
            min_threshold = min_p * max_prob
            logits_temp[probs < min_threshold] = float('-inf')

            # Top-p
            probs_filtered = torch.softmax(logits_temp, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs_filtered, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[:, 0] = False
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()

            # Sample
            next_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(1, next_idx)

            token_id = next_token.item()
            predicted.append(token_id)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check EOS
            if token_id == t3.hp.stop_speech_token:
                print(f"EOS detected at step {i+1}")
                break

            # Progress
            if (i + 1) % 20 == 0:
                eos_prob = probs[0, t3.hp.stop_speech_token].item()
                print(f"Step {i+1}: token={token_id}, EOS_prob={eos_prob:.6f}")

            # Next embedding
            next_emb = t3.speech_emb(next_token)
            next_emb = next_emb + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_emb = torch.cat([next_emb, next_emb], dim=0)

            # Forward with cache
            output = t3.tfmr(
                inputs_embeds=next_emb,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            past = output.past_key_values

    n_tokens = len(predicted)
    print(f"\nGenerated {n_tokens} speech tokens")
    print(f"Approximate duration: {n_tokens / 86:.2f} seconds")
    if n_tokens > 0:
        print(f"First 10: {predicted[:10]}")
        print(f"Last 10: {predicted[-10:]}")

    # Check special tokens
    print(f"\nSpecial tokens:")
    print(f"  start_speech_token: {t3.hp.start_speech_token}")
    print(f"  stop_speech_token: {t3.hp.stop_speech_token}")

    # Save
    safetensors.torch.save_file({
        "generated_tokens": torch.tensor(predicted),
    }, "/outputs/python_generated.safetensors")
    print("\nSaved to /outputs/python_generated.safetensors")

if __name__ == "__main__":
    main()
