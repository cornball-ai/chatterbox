import torch
import gc

def vram(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{label:<55}  alloc={alloc:.3f}GB  reserved={reserved:.3f}GB  peak={peak:.3f}GB")

# Load model
print("=" * 90)
print("MEMORY PROFILE: Python chatterbox S3Gen")
print("=" * 90)

from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained("cuda")
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
vram("Model loaded + empty_cache")

# Prepare conditionals
model.prepare_conditionals("/tmp/ref.wav")
gc.collect(); torch.cuda.empty_cache()
vram("After prepare_conditionals")

# Generation 1
print("\n--- Generation 1 ---")
torch.cuda.reset_peak_memory_stats()
vram("Before generate 1")
audio1 = model.generate("The quick brown fox jumps over the lazy dog.")
gc.collect(); torch.cuda.empty_cache()
vram("After generate 1 + empty_cache")

# Generation 2
print("\n--- Generation 2 ---")
torch.cuda.reset_peak_memory_stats()
vram("Before generate 2")
audio2 = model.generate("This is a second sentence to test memory accumulation.")
gc.collect(); torch.cuda.empty_cache()
vram("After generate 2 + empty_cache")

# Generation 3
print("\n--- Generation 3 ---")
torch.cuda.reset_peak_memory_stats()
vram("Before generate 3")
audio3 = model.generate("And a third sentence to check for leaks.")
gc.collect(); torch.cuda.empty_cache()
vram("After generate 3 + empty_cache")

# Generation 4
print("\n--- Generation 4 ---")
torch.cuda.reset_peak_memory_stats()
vram("Before generate 4")
audio4 = model.generate("Finally a fourth sentence for good measure.")
gc.collect(); torch.cuda.empty_cache()
vram("After generate 4 + empty_cache")

# Delete audio refs and see if memory drops
del audio1, audio2, audio3, audio4
gc.collect(); torch.cuda.empty_cache()
vram("After deleting all audio + empty_cache")

print("\n" + "=" * 90)
print("DETAILED: Isolating S3Gen from T3")
print("=" * 90)

# Now isolate S3Gen memory usage
torch.cuda.reset_peak_memory_stats()
gc.collect(); torch.cuda.empty_cache()
vram("Clean state")

with torch.inference_mode():
    # Run T3 to get tokens
    import torch.nn.functional as F
    from chatterbox.tts import punc_norm
    from chatterbox.models.s3tokenizer import drop_invalid_tokens

    text = punc_norm("Hello world, this is a test of memory usage.")
    text_tokens = model.tokenizer.text_to_tokens(text).to("cuda")
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    vram("After text tokenization")

    speech_tokens = model.t3.inference(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=1000,
        temperature=0.8,
        cfg_weight=0.5,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    )
    vram("After T3 inference")

    speech_tokens = speech_tokens[0]
    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens = speech_tokens[speech_tokens < 6561]
    speech_tokens = speech_tokens.to("cuda")
    print(f"  Speech tokens shape: {speech_tokens.shape}")

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram("Before S3Gen (after T3 cleanup)")

    # Run S3Gen
    wav, _ = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
    )
    vram("After S3Gen inference")

    gc.collect(); torch.cuda.empty_cache()
    vram("After S3Gen + empty_cache")

    # Run S3Gen again
    torch.cuda.reset_peak_memory_stats()
    vram("Before 2nd S3Gen")
    wav2, _ = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
    )
    vram("After 2nd S3Gen")

    gc.collect(); torch.cuda.empty_cache()
    vram("After 2nd S3Gen + empty_cache")

    # Run S3Gen 3rd time
    torch.cuda.reset_peak_memory_stats()
    vram("Before 3rd S3Gen")
    wav3, _ = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
    )
    vram("After 3rd S3Gen")

    gc.collect(); torch.cuda.empty_cache()
    vram("After 3rd S3Gen + empty_cache")

    del wav, wav2, wav3
    gc.collect(); torch.cuda.empty_cache()
    vram("After deleting all wavs")

print("\n" + "=" * 90)
print("SOLVE_EULER INTERNALS")
print("=" * 90)

# Monkey-patch solve_euler to track memory at each step
import types

def profiled_solve_euler(self, x, t_span, mu, mask, spks, cond):
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(dim=0)
    sol = []

    x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
    mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
    mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
    t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
    spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
    cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)

    vram(f"  solve_euler: after allocating _in tensors")

    for step in range(1, len(t_span)):
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t.unsqueeze(0)
        spks_in[0] = spks
        cond_in[0] = cond
        dphi_dt = self.forward_estimator(
            x_in, mask_in, mu_in, t_in, spks_in, cond_in
        )
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
        vram(f"  solve_euler step {step}/10")

    vram(f"  solve_euler: before return (sol has {len(sol)} items)")
    return sol[-1].float()

model.s3gen.flow.decoder.solve_euler = types.MethodType(profiled_solve_euler, model.s3gen.flow.decoder)

with torch.inference_mode():
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram("Before profiled S3Gen")

    wav_prof, _ = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
    )
    vram("After profiled S3Gen")

    gc.collect(); torch.cuda.empty_cache()
    vram("After profiled S3Gen + empty_cache")

print("\n" + "=" * 90)
print("SUMMARY OF PYTHON MEMORY MANAGEMENT PATTERNS")
print("=" * 90)
print("""
1. Python does NOT call torch.cuda.empty_cache() anywhere in the codebase.
2. Python does NOT use 'del' for intermediate tensors in S3Gen/CFM.
3. Python solve_euler stores ALL 10 steps in sol[] list, only returns sol[-1].
4. Python pre-allocates _in tensors and reuses via in-place assignment (x_in[:] = x).
5. Python uses @torch.inference_mode() which disables autograd tracking.
6. CausalConditionalCFM uses pre-allocated rand_noise buffer (50*300 frames).
7. Memory management relies entirely on Python GC + PyTorch CUDA allocator.
""")
