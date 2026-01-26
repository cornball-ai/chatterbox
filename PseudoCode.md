# ═══════════════════════════════════════════════════════════════════════════════
# CHATTERBOX TTS - COMPLETE PIPELINE (English v0.1.4)
# ═══════════════════════════════════════════════════════════════════════════════

text <- "Hello world!"
reference_audio <- "speaker.wav"

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE ENCODER - extracts speaker embedding from reference audio
# Uses 40-mel spectrogram at 16kHz
# ═══════════════════════════════════════════════════════════════════════════════

# Voice encoder mel: 40 mels, 16kHz, n_fft=400, hop=160
ve_mel <- mel_spectrogram_ve(reference_audio)    # (1, time, 40) for LSTM input
#         Parameters: n_mels=40, sr=16000, n_fft=400, hop=160, fmin=0, fmax=8000
#         Uses Slaney mel scale (librosa default), power=2, no log compression

speaker_emb <- voice_encoder_lstm(ve_mel)        # (1, 256) - WHO is speaking
#              3-layer LSTM → proj → ReLU → L2 normalize
#              Processes overlapping 160-frame partials, averages embeddings

# ═══════════════════════════════════════════════════════════════════════════════
# S3 TOKENIZER - extracts prompt speech tokens from reference audio
# Uses Whisper-style encoder + FSQ quantization
# CORRECTED: Uses 128 mels at 16kHz (not 80 mels at 24kHz)
# ═══════════════════════════════════════════════════════════════════════════════

# S3 tokenizer config:
#   n_mels: 128, n_fft: 400, sample_rate: 16kHz
#   encoder: AudioEncoderV2 (Whisper-style, 6 layers, 20 heads, 1280 dim)
#   quantizer: FSQVectorQuantization (codebook_size: 6561)
#   output: 25 tokens/second (frame rate)
prompt_speech_tokens <- s3_tokenizer(reference_audio)  # (1, 150) - max_len capped

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: T3 - Autoregressive text → speech tokens
# (No diffusion analog - this is like GPT generating discrete codes)
# ═══════════════════════════════════════════════════════════════════════════════

# --- Text tokenization ---
text_tokens <- bpe_tokenizer(text)                # "Hello world!" → [15496, 995, ...]
text_tokens <- c(SOT_TOKEN, text_tokens, EOT_TOKEN)  # Add start/end of text

# --- Build 34-position conditioning sequence ---
# Position [1]: Speaker identity
cond_speaker <- linear_256_to_1024(speaker_emb)   # (1, 1, 1024)

# Positions [2-33]: Perceiver-compressed reference speech
prompt_emb <- speech_embedding(prompt_speech_tokens)  # (1, ~150, 1024)
cond_perceiver <- perceiver_resampler(prompt_emb)     # (1, 32, 1024)
#                 └─ 32 learned queries cross-attend to ~150 tokens
#                    then self-attend, compressing variable-length
#                    reference into fixed 32 positions

# Position [34]: Emotion/exaggeration control
cond_emotion <- emotion_linear(exaggeration)      # (1, 1, 1024)

# Concatenate conditioning
cond <- concat(cond_speaker, cond_perceiver, cond_emotion)  # (1, 34, 1024)
cond_uncond <- zeros_like(cond)                             # For CFG

# --- Classifier-Free Guidance setup (batch=2) ---
cond_cfg <- stack(cond, cond_uncond)              # (2, 34, 1024)
text_emb <- text_embedding(text_tokens)           # (1, seq, 1024)
text_emb_cfg <- stack(text_emb, text_emb)         # (2, seq, 1024)

# --- Autoregressive generation loop ---
speech_tokens <- c(BOS_TOKEN)                     # start_speech_token
kv_cache <- NULL

for (step in 1:max_tokens) {
  # Build input sequence: [conditioning | text | speech_so_far]
  speech_emb <- speech_embedding(speech_tokens) + position_embedding(speech_tokens)
  speech_emb_cfg <- stack(speech_emb, speech_emb)           # (2, speech_len, 1024)

  input_embeds <- concat(cond_cfg, text_emb_cfg, speech_emb_cfg)  # (2, total_len, 1024)

  # Llama transformer forward (with KV cache after first step)
  hidden, kv_cache <- llama_transformer(input_embeds, kv_cache)   # (2, total_len, 1024)

  # Get logits from last position
  logits <- speech_head(hidden[, -1, ])           # (2, vocab_size=8194)

  # --- CFG: amplify conditioning effect ---
  cond_logits <- logits[1, ]                      # Conditioned path
  uncond_logits <- logits[2, ]                    # Unconditioned path
  logits <- cond_logits + cfg_weight * (cond_logits - uncond_logits)

  # --- Sampling with temperature, min-p, top-p ---
  logits <- logits / temperature
  probs <- softmax(logits)

  # Min-p: remove tokens below (min_p * max_prob)
  logits[probs < min_p * max(probs)] <- -Inf
  probs <- softmax(logits)                        # Recompute after filtering

  # Top-p (nucleus): keep smallest set summing to p
  sorted_result <- sort(probs, decreasing = TRUE, index.return = TRUE)
  sorted_probs <- sorted_result$x
  sorted_idx <- sorted_result$ix                  # NOTE: R returns 1-indexed
  cumsum_probs <- cumsum(sorted_probs)
  cutoff <- which(cumsum_probs > top_p)[1]
  if (!is.na(cutoff) && cutoff > 1) {
    sorted_probs[(cutoff):length(sorted_probs)] <- 0
  }
  sorted_probs <- sorted_probs / sum(sorted_probs)

  # Sample from filtered distribution
  sample_idx <- sample(seq_along(sorted_probs), size = 1, prob = sorted_probs)
  next_token <- sorted_idx[sample_idx]            # 1-indexed token position

  # NOTE: For EOS check, convert to 0-indexed token ID
  token_id <- next_token - 1L                     # R torch_sort returns 1-indexed
  if (token_id == stop_speech_token) break

  speech_tokens <- c(speech_tokens, next_token)
}

# speech_tokens ≈ 400-800 discrete codes (86 tokens/second of audio)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: S3Gen - Speech tokens → mel spectrogram (Flow Matching)
# (This parallels diffusion: start with noise, iteratively denoise)
# Uses 80-mel spectrogram at 24kHz
# ═══════════════════════════════════════════════════════════════════════════════

# --- Encode speech tokens ---
h <- s3gen_encoder(speech_tokens)                 # (1, 80, time)

# --- Project speaker embedding for CFM ---
spk_emb <- normalize(speaker_emb)
spk_emb <- linear_256_to_80(spk_emb)              # (1, 80)
spk_expanded <- expand_to_time(spk_emb, time)     # (1, 80, time)

# --- Prompt conditioning (mel from reference audio) ---
# S3Gen uses 80-mel at 24kHz (different from voice encoder's 40-mel at 16kHz)
prompt_mel <- mel_spectrogram_s3gen(reference_audio)  # (1, 80, prompt_time)

# --- Flow Matching: ODE from noise to mel ---
noisy_mel <- rnorm(1, 80, time)                   # Start with pure noise

timesteps <- seq(0, 1, length.out = 10)           # 10 Euler steps
dt <- 1 / 10

for (t in timesteps) {
  # CFM estimator predicts velocity field
  # Input channels: [noisy_mel | encoder_h | speaker | prompt_mel]
  #                     80     +    80     +   80    +    80    = 320

  velocity <- cfm_estimator(
    x = noisy_mel,                                # Current state
    mu = h,                                       # Encoder output (target info)
    spks = spk_expanded,                          # Speaker identity
    cond = prompt_mel,                            # Reference style
    t = t                                         # Timestep embedding
  )

  # Euler integration step
  noisy_mel <- noisy_mel + velocity * dt
}

clean_mel <- noisy_mel                            # (1, 80, time) - denoised mel

# ═══════════════════════════════════════════════════════════════════════════════
# VOCODER: HiFi-GAN mel → waveform
# (Like VAE decoder: latent representation → final output)
# ═══════════════════════════════════════════════════════════════════════════════

waveform <- hifigan_vocoder(clean_mel)            # (1, samples) @ 24kHz

write_wav("output.wav", waveform, sample_rate = 24000)

# ═══════════════════════════════════════════════════════════════════════════════
# KEY IMPLEMENTATION NOTES FOR R TORCH
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. TWO DIFFERENT MEL SPECTROGRAMS:
#    - Voice encoder: 40 mels, 16kHz, n_fft=400, hop=160, Slaney scale
#    - S3Gen/S3Tokenizer: 80 mels, 24kHz
#
# 2. R TORCH 1-INDEXING:
#    - torch_sort() returns 1-indexed indices (Python returns 0-indexed)
#    - nn_embedding() expects 1-indexed inputs
#    - When comparing token IDs to Python constants, subtract 1
#
# 3. FLOAT16 DTYPE PRESERVATION:
#    - R scalar ops promote tensors through float64
#    - Use tensor methods: x$mul(scale) not x * scale
#    - Use explicit dtype: torch_tensor(val, dtype = tensor$dtype)
#
# 4. MEL FILTERBANK:
#    - Use Slaney formula (librosa default), NOT HTK
#    - Slaney: linear below 1000 Hz, log above
#    - Normalize by Hz bandwidth, not mel bandwidth
#
# 5. STFT PADDING:
#    - librosa center=TRUE pads n_fft // 2 on each side
#    - NOT (n_fft - hop_size) / 2
