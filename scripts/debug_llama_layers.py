#!/usr/bin/env python3
"""Extract Llama layer-by-layer outputs for comparison with R implementation."""

import torch
import safetensors.torch
import sys

from chatterbox.tts import ChatterboxTTS
import torch.nn.functional as F

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading Chatterbox model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    t3 = model.t3

    # Use JFK reference audio
    ref_audio = "/ref_audio/jfk.wav"

    # Simple test text
    text = "Hello world"

    # Prepare conditionals
    print("Preparing conditionals...")
    model.prepare_conditionals(ref_audio, exaggeration=0.5)
    t3_cond = model.conds.t3

    # Tokenize text
    text_tokens = model.tokenizer.text_to_tokens(text).to(device)
    eot = t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    # Double for CFG
    cfg_weight = 0.5
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)

    # BOS token
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_cfg = torch.cat([bos, bos], dim=0)

    print(f"Text tokens shape: {text_tokens_cfg.shape}")
    print(f"Text: {text}")

    # Prepare input embeddings
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens_cfg,
        speech_tokens=bos_cfg,
        cfg_weight=cfg_weight,
    )

    print(f"Input embeds shape: {embeds.shape}")
    print(f"Conditioning length: {len_cond}")

    # Collect outputs
    outputs = {}

    # Save input embeds
    outputs["input_embeds"] = embeds.detach().cpu().float()
    outputs["len_cond"] = torch.tensor([len_cond])

    # Save text tokens for R to use
    outputs["text_tokens"] = text_tokens_cfg.detach().cpu()

    # Save speaker embedding
    outputs["speaker_emb"] = t3_cond.speaker_emb.detach().cpu().float()

    # Hook to capture layer outputs
    layer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            layer_outputs[name] = out.detach().clone()
        return hook

    # Register hooks on each Llama layer
    # Python structure: t3.tfmr.layers (not t3.tfmr.model.layers)
    hooks = []
    for i, layer in enumerate(t3.tfmr.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

    # Hook on final norm
    hooks.append(t3.tfmr.norm.register_forward_hook(make_hook("final_norm")))

    # Forward pass through Llama
    with torch.inference_mode():
        output = t3.tfmr(
            inputs_embeds=embeds,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save layer outputs
    for name, tensor in layer_outputs.items():
        outputs[name] = tensor.detach().cpu().float()

    # Save final hidden state
    outputs["last_hidden_state"] = output.last_hidden_state.detach().cpu().float()

    # Get logits (clone to avoid inference mode issues)
    last_hidden = output.last_hidden_state[:, -1, :].clone()
    logits = t3.speech_head(last_hidden)
    outputs["logits"] = logits.detach().cpu().float()

    # Print summary
    print("\n=== Layer-by-layer statistics ===")
    for name in sorted(layer_outputs.keys(), key=lambda x: (0 if x == "final_norm" else 1, x)):
        t = layer_outputs[name]
        # Get stats for cond path (first batch element)
        t_cond = t[0]
        print(f"{name}: shape={list(t.shape)}, cond_mean={t_cond.mean().item():.6f}, cond_std={t_cond.std().item():.6f}")

    print(f"\nLast hidden (cond) mean: {output.last_hidden_state[0].mean().item():.6f}")
    print(f"Last hidden (cond) std: {output.last_hidden_state[0].std().item():.6f}")
    print(f"Logits (cond) mean: {logits[0].mean().item():.6f}")
    print(f"Logits (cond) std: {logits[0].std().item():.6f}")

    # Save
    output_path = "/outputs/llama_layers.safetensors"
    safetensors.torch.save_file(outputs, output_path)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
