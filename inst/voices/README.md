# The built-in voice

Python chatterbox's default voice is `conds.pt` in each model repository:
the conditionals `prepare_conditionals()` computed once from a reference
clip and saved with `torch.save()`. R torch cannot read that nested
pickle, so the same tensors are here as safetensors, one file per model
set, read by `builtin_voice()`.

| File | Source | Source sha256 | Tensors |
|---|---|---|---|
| `chatterbox.safetensors` (105,396 B) | `ResembleAI/chatterbox`, `conds.pt` at revision `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | `6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e` | `t3.speaker_emb` F32 [1,256]; `t3.cond_prompt_speech_tokens` I64 [1,150]; `t3.emotion_adv` F32 [1,1,1]; `gen.prompt_token` I64 [1,157]; `gen.prompt_token_len` I64 [1]; `gen.prompt_feat` F32 [1,314,80]; `gen.embedding` F32 [1,192] |
| `chatterbox-turbo.safetensors` (167,468 B) | `ResembleAI/chatterbox-turbo`, `conds.pt` at revision `749d1c1a46eb10492095d68fbcf55691ccf137cd` | `b1852099306fd6a7814eb9d0bd10186caba7249596cc23868f78a0eefbfa5033` | the same keys with [1,375], [1,250] and [1,500,80] |

The conversion (2026-11-02) opened each zip pickle with an unpickler that
admits only `torch._utils._rebuild_tensor_v2`, the storage classes and
`OrderedDict`, copied each tensor's bytes out of its storage in row-major
order (`gen.prompt_feat` is saved as a transposed view and
`gen.prompt_token` as a view one column short of its storage), and wrote
them unchanged, little-endian, with the keys above. Entries that are
`None` in the pickle (`t3.clap_emb`, `t3.cond_prompt_speech_emb`,
`gen.prompt_feat_len`) are not written; `builtin_voice()` sets
`prompt_feat_len` to `NULL` as `create_voice_embedding()` does.
`t3.emotion_adv` is kept for the record; `generate()` takes exaggeration
from its own argument.

Both model repositories are published by Resemble AI under the MIT
license, which covers these files as it covers the weights.
