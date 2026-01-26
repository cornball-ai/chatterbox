#!/usr/bin/env python3
"""Trace CFM estimator details including block structure."""

import torch
from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

print("\n=== Estimator Dimensions ===")

# Time embeddings
time_emb = estimator.time_embeddings
print(f"time_embeddings: SinusoidalPosEmb")
# Check input/output by test
with torch.no_grad():
    t = torch.tensor([0.5], device='cuda')
    t_emb = time_emb(t)
    print(f"  input: {t.shape} -> output: {t_emb.shape}")

# Time MLP
time_mlp = estimator.time_mlp
print(f"time_mlp: {time_mlp.linear_1.in_features} -> {time_mlp.linear_1.out_features} -> {time_mlp.linear_2.out_features}")

# Down blocks
print(f"\n=== Down Blocks ({len(estimator.down_blocks)}) ===")
for i, block in enumerate(estimator.down_blocks):
    resnet, transformers, downsample = block
    print(f"down_block[{i}]:")
    print(f"  resnet: {resnet.res_conv.weight.shape}")
    print(f"  transformers: {len(transformers)} BasicTransformerBlocks")
    if hasattr(transformers[0], 'attn1'):
        attn1 = transformers[0].attn1
        print(f"    attn1 heads: {attn1.heads}, dim: {attn1.to_q.in_features}")
    print(f"  downsample: {downsample.weight.shape}")

# Mid blocks
print(f"\n=== Mid Blocks ({len(estimator.mid_blocks)}) ===")
for i, block in enumerate(estimator.mid_blocks):
    resnet, transformers = block
    print(f"mid_block[{i}]:")
    print(f"  resnet: {resnet.res_conv.weight.shape}")
    print(f"  transformers: {len(transformers)} BasicTransformerBlocks")

# Up blocks
print(f"\n=== Up Blocks ({len(estimator.up_blocks)}) ===")
for i, block in enumerate(estimator.up_blocks):
    resnet, transformers, upsample = block
    print(f"up_block[{i}]:")
    print(f"  resnet: {resnet.res_conv.weight.shape}")
    print(f"  transformers: {len(transformers)} BasicTransformerBlocks")
    print(f"  upsample: {upsample.weight.shape}")

# Final
print(f"\n=== Final ===")
print(f"final_block.block: CausalConv1d {estimator.final_block.block[0].weight.shape}")
print(f"final_proj: Conv1d {estimator.final_proj.weight.shape}")

# Check CausalResnetBlock1D structure
print("\n=== CausalResnetBlock1D Structure ===")
resnet = estimator.down_blocks[0][0]
print(f"Type: {type(resnet).__name__}")
print(f"mlp: {resnet.mlp}")
print(f"block1: {type(resnet.block1).__name__}")
print(f"  conv: {resnet.block1.block[0].weight.shape}")
print(f"block2: {type(resnet.block2).__name__}")
print(f"  conv: {resnet.block2.block[0].weight.shape}")
print(f"res_conv: {resnet.res_conv.weight.shape}")

# Check BasicTransformerBlock structure
print("\n=== BasicTransformerBlock Structure ===")
tfm = estimator.down_blocks[0][1][0]
print(f"Type: {type(tfm).__name__}")
for name, child in tfm.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'weight') and child.weight is not None:
        print(f"    weight: {child.weight.shape}")

# Get shape flow through estimator
print("\n=== Shape Flow Through Estimator ===")
with torch.no_grad():
    batch = 1
    time = 100

    x = torch.randn(batch, 80, time, device='cuda')
    mask = torch.ones(batch, 1, time, device='cuda')
    mu = torch.randn(batch, 80, time, device='cuda')
    t = torch.ones(batch, device='cuda') * 0.5
    spks = torch.randn(batch, 80, device='cuda')
    cond = torch.randn(batch, 80, time, device='cuda')

    print(f"Input x: {x.shape}")
    print(f"Input mu: {mu.shape}")
    print(f"Input spks: {spks.shape}")
    print(f"Input cond: {cond.shape}")

    # Pack inputs
    from einops import pack, repeat
    x_packed = pack([x, mu], "b * t")[0]
    print(f"After packing x+mu: {x_packed.shape}")

    spks_exp = repeat(spks, "b c -> b c t", t=time)
    x_packed = pack([x_packed, spks_exp], "b * t")[0]
    print(f"After packing spks: {x_packed.shape}")

    x_packed = pack([x_packed, cond], "b * t")[0]
    print(f"After packing cond: {x_packed.shape}")

    # Time embedding
    t_emb = estimator.time_embeddings(t).to(t.dtype)
    t_emb = estimator.time_mlp(t_emb)
    print(f"Time embedding: {t_emb.shape}")
