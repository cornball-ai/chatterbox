#!/usr/bin/env python3
"""Trace CFM decoder for R comparison."""

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from chatterbox.tts import ChatterboxTTS
import inspect

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
s3gen = model.s3gen
flow = s3gen.flow
decoder = flow.decoder

# ============================================================================
# Step 1: CausalConditionalCFM architecture
# ============================================================================
print("\n=== Step 1: CausalConditionalCFM Architecture ===")
print(f"decoder type: {type(decoder).__name__}")

print("\ndecoder children:")
for name, child in decoder.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in list(child.named_children())[:3]:
            print(f"    {n2}: {type(c2).__name__}")

# Get forward/call source
print("\n=== Step 2: CausalConditionalCFM Forward ===")
try:
    fwd_source = inspect.getsource(type(decoder).forward)
    print("decoder.forward source:")
    print(fwd_source)
except Exception as e:
    print(f"Error: {e}")

# Check __call__
try:
    call_source = inspect.getsource(type(decoder).__call__)
    print("\ndecoder.__call__ source:")
    print(call_source)
except Exception as e:
    print(f"Error getting __call__: {e}")

# ============================================================================
# Step 3: ConditionalDecoder (estimator)
# ============================================================================
print("\n=== Step 3: ConditionalDecoder (estimator) ===")
estimator = decoder.estimator
print(f"estimator type: {type(estimator).__name__}")

print("\nestimator children:")
for name, child in estimator.named_children():
    print(f"  {name}: {type(child).__name__}")

try:
    est_source = inspect.getsource(type(estimator).forward)
    print("\nestimator.forward source:")
    print(est_source)
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Step 4: Flow parameters
# ============================================================================
print("\n=== Step 4: Flow Parameters ===")
print(f"flow.input_embedding: {flow.input_embedding}")
print(f"  num_embeddings: {flow.input_embedding.num_embeddings}")
print(f"  embedding_dim: {flow.input_embedding.embedding_dim}")
print(f"flow.spk_embed_affine_layer: {flow.spk_embed_affine_layer}")
print(f"  in_features: {flow.spk_embed_affine_layer.in_features}")
print(f"  out_features: {flow.spk_embed_affine_layer.out_features}")

# Check encoder
print(f"\nflow.encoder type: {type(flow.encoder).__name__}")
print(f"flow.encoder_proj: {flow.encoder_proj}")
print(f"  in_features: {flow.encoder_proj.in_features}")
print(f"  out_features: {flow.encoder_proj.out_features}")

# Check decoder params
if hasattr(decoder, 'sigma'):
    print(f"\ndecoder.sigma: {decoder.sigma}")
if hasattr(flow, 'output_size'):
    print(f"flow.output_size: {flow.output_size}")
if hasattr(flow, 'token_mel_ratio'):
    print(f"flow.token_mel_ratio: {flow.token_mel_ratio}")
if hasattr(flow, 'pre_lookahead_len'):
    print(f"flow.pre_lookahead_len: {flow.pre_lookahead_len}")

# ============================================================================
# Step 5: UpsampleConformerEncoder
# ============================================================================
print("\n=== Step 5: UpsampleConformerEncoder ===")
encoder = flow.encoder
print(f"encoder type: {type(encoder).__name__}")

print("\nencoder children:")
for name, child in encoder.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        sub = list(child.named_children())[:3]
        for n2, c2 in sub:
            print(f"    {n2}: {type(c2).__name__}")

# Get encoder params
if hasattr(encoder, 'output_size'):
    print(f"\nencoder.output_size: {encoder.output_size()}")

# ============================================================================
# Step 6: Run flow with actual data and save intermediates
# ============================================================================
print("\n=== Step 6: Run Flow with Intermediates ===")
ref = load_file("/outputs/mel_reference.safetensors")
wav_np = ref["audio_wav"].numpy()
ref_wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).float().cuda()

torch.manual_seed(42)
test_tokens = torch.randint(0, 6561, (1, 31), device='cuda')
token_len = torch.LongTensor([31]).cuda()

with torch.inference_mode():
    # Get ref embedding
    ref_dict = s3gen.embed_ref(ref_wav_tensor, 16000)
    print(f"ref_dict keys: {ref_dict.keys()}")

    prompt_token = ref_dict['prompt_token']
    prompt_token_len = ref_dict['prompt_token_len']
    prompt_feat = ref_dict['prompt_feat']
    embedding = ref_dict['embedding']

    print(f"\nInputs:")
    print(f"  test_tokens: {test_tokens.shape}")
    print(f"  prompt_token: {prompt_token.shape}")
    print(f"  prompt_feat: {prompt_feat.shape}")
    print(f"  embedding: {embedding.shape}")

    # Manual flow.inference trace
    print("\nTracing flow.inference manually:")

    # 1. Normalize and project embedding
    import torch.nn.functional as F
    emb_norm = F.normalize(embedding, dim=1)
    spk_emb = flow.spk_embed_affine_layer(emb_norm)
    print(f"  spk_emb: {spk_emb.shape}")

    # 2. Concat tokens
    all_tokens = torch.concat([prompt_token, test_tokens], dim=1)
    all_token_len = prompt_token_len + token_len
    print(f"  all_tokens: {all_tokens.shape}")
    print(f"  all_token_len: {all_token_len}")

    # 3. Create mask and embed
    # Define make_pad_mask inline
    def make_pad_mask(lengths, max_len=None):
        batch_size = lengths.size(0)
        max_len = max_len if max_len else lengths.max().item()
        seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask

    mask = (~make_pad_mask(all_token_len)).unsqueeze(-1).to(spk_emb)
    token_emb = flow.input_embedding(torch.clamp(all_tokens, min=0, max=flow.input_embedding.num_embeddings-1)) * mask
    print(f"  token_emb: {token_emb.shape}")

    # 4. Encode
    h, h_lengths = flow.encoder(token_emb, all_token_len)
    print(f"  encoder output h: {h.shape}")

    # 5. Trim (finalize=False)
    pre_lookahead = flow.pre_lookahead_len * flow.token_mel_ratio
    h_trimmed = h[:, :-pre_lookahead]
    print(f"  h_trimmed (no lookahead): {h_trimmed.shape}")

    # 6. Project
    mel_len1 = prompt_feat.shape[1]
    mel_len2 = h_trimmed.shape[1] - mel_len1
    print(f"  mel_len1 (prompt): {mel_len1}")
    print(f"  mel_len2 (generated): {mel_len2}")

    h_proj = flow.encoder_proj(h_trimmed)
    print(f"  h_proj: {h_proj.shape}")

    # 7. Build conditions
    conds = torch.zeros([1, mel_len1 + mel_len2, flow.output_size], device='cuda').to(h_proj.dtype)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)
    print(f"  conds: {conds.shape}")

    # 8. Run decoder
    dec_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h_proj)
    print(f"  dec_mask: {dec_mask.shape}")

    mu = h_proj.transpose(1, 2).contiguous()
    print(f"  mu (decoder input): {mu.shape}")

    feat, _ = decoder(
        mu=mu,
        mask=dec_mask.unsqueeze(1),
        spks=spk_emb,
        cond=conds,
        n_timesteps=10
    )
    print(f"  decoder output: {feat.shape}")

    # 9. Extract generated part
    feat_gen = feat[:, :, mel_len1:]
    print(f"  feat_gen (generated mel): {feat_gen.shape}")

# Save intermediates
print("\n=== Step 7: Save Intermediates ===")
save_dict = {
    "test_tokens": test_tokens.cpu().contiguous(),
    "prompt_token": prompt_token.cpu().contiguous(),
    "prompt_feat": prompt_feat.cpu().contiguous(),
    "embedding": embedding.cpu().contiguous(),
    "spk_emb": spk_emb.cpu().contiguous(),
    "all_tokens": all_tokens.cpu().contiguous(),
    "token_emb": token_emb.cpu().contiguous(),
    "encoder_h": h.cpu().contiguous(),
    "h_proj": h_proj.cpu().contiguous(),
    "conds": conds.cpu().contiguous(),
    "mu": mu.cpu().contiguous(),
    "decoder_output": feat.cpu().contiguous(),
    "feat_gen": feat_gen.cpu().contiguous(),
}
save_file(save_dict, "/outputs/cfm_steps.safetensors")
print(f"Saved to /outputs/cfm_steps.safetensors")
print(f"Keys: {list(save_dict.keys())}")
