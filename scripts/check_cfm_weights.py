#!/usr/bin/env python3
"""Check CFM estimator weight structure."""

from chatterbox.tts import ChatterboxTTS

print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained("cuda")
estimator = model.s3gen.flow.decoder.estimator

print("\n=== State Dict Keys ===")
state_dict = estimator.state_dict()
keys = sorted(state_dict.keys())

# Group by prefix
prefixes = {}
for key in keys:
    prefix = key.split('.')[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(key)

for prefix in sorted(prefixes.keys()):
    print(f"\n{prefix} ({len(prefixes[prefix])} keys):")
    for key in prefixes[prefix][:10]:
        shape = tuple(state_dict[key].shape)
        print(f"  {key}: {shape}")
    if len(prefixes[prefix]) > 10:
        print(f"  ... and {len(prefixes[prefix]) - 10} more")

# Print total params
total = sum(p.numel() for p in estimator.parameters())
print(f"\nTotal parameters: {total:,}")

# Check FeedForward structure
print("\n=== FeedForward Structure ===")
ff = estimator.down_blocks[0][1][0].ff
for name, child in ff.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2).__name__}")
            if hasattr(c2, 'weight') and c2.weight is not None:
                print(f"      weight: {c2.weight.shape}")

# Check attention structure
print("\n=== Attention Structure ===")
attn = estimator.down_blocks[0][1][0].attn1
for name, child in attn.named_children():
    print(f"  {name}: {type(child).__name__}")
    if hasattr(child, 'weight') and child.weight is not None:
        print(f"    weight: {child.weight.shape}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2).__name__}")
            if hasattr(c2, 'weight') and c2.weight is not None:
                print(f"      weight: {c2.weight.shape}")
