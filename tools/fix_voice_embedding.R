#!/usr/bin/env Rscript --vanilla
# Voice-embedding stage fixture: the full torch conditioning chain
# (create_voice_embedding incl. embed_ref) on jfk.wav. Torch only (CPU).
library(chatterbox)

model <- chatterbox(device = "cpu")
aud <- chatterbox:::read_audio("inst/audio/jfk.wav")
ve <- torch::with_no_grad({
  create_voice_embedding(model, "inst/audio/jfk.wav")
})

paths <- chatterbox:::get_model_paths()
dir.create("tools/fixtures", showWarnings = FALSE, recursive = TRUE)
saveRDS(list(
  samples = aud$samples, sr = aud$sr,
  ve_embedding = as.array(ve$ve_embedding$cpu()),
  cond_prompt_tokens = as.array(ve$cond_prompt_speech_tokens$cpu()),
  prompt_token = as.array(ve$ref_dict$prompt_token$cpu()),
  prompt_feat = as.array(ve$ref_dict$prompt_feat$cpu()),
  xvector = as.array(ve$ref_dict$embedding$cpu()),
  ve_path = paths$ve, s3gen_path = paths$s3gen),
  "tools/fixtures/voice_embedding.rds")
cat(sprintf(
  "voice_embedding fixture: sr %d, ve %s, cond_tokens %s, prompt_token %s, prompt_feat %s, xvector %s\n",
  aud$sr,
  paste(dim(as.array(ve$ve_embedding)), collapse = "x"),
  paste(dim(as.array(ve$cond_prompt_speech_tokens)), collapse = "x"),
  paste(dim(as.array(ve$ref_dict$prompt_token)), collapse = "x"),
  paste(dim(as.array(ve$ref_dict$prompt_feat)), collapse = "x"),
  paste(dim(as.array(ve$ref_dict$embedding)), collapse = "x")))
