# anvl/yunque e2e glue: the voice-embedding (reference conditioning) stage
# and the generation stage. Orchestrates the ported frontends + encoders
# exactly as the torch side's create_voice_embedding /
# compute_speaker_embedding / compute_xvector_embedding / s3gen$embed_ref /
# generate chain does.

# T3 config constants the generation glue needs (t3_config_english).
.yq_t3_config <- function() {
  list(start_text_token = 255L, stop_text_token = 0L,
    start_speech_token = 6561L, stop_speech_token = 6562L,
    max_text_tokens = 2048L, speech_cond_prompt_len = 150L)
}

# Verbatim pure-R copy of trim_silence (audio_utils.R, librosa.effects.trim
# semantics) so the anvl path stands alone.
.yq_trim_silence <- function(samples, top_db = 20, frame_length = 2048L,
                             hop_length = 512L) {
  n <- length(samples)
  if (n == 0) {
    return(samples)
  }
  pad <- frame_length %/% 2L
  padded <- c(rep(0, pad), samples, rep(0, pad))
  n_frames <- 1L + (length(padded) - frame_length) %/% hop_length
  power <- vapply(seq_len(n_frames), function(i) {
    s <- (i - 1L) * hop_length
    mean(padded[(s + 1L):(s + frame_length)]^2)
  }, numeric(1))
  ref <- max(power)
  if (ref <= 0) {
    return(samples)
  }
  db <- 10 * log10(pmax(power, 1e-10) / ref)
  nonsilent <- which(db > -top_db)
  if (length(nonsilent) == 0) {
    return(samples[0])
  }
  start <- (nonsilent[1] - 1L) * hop_length
  end <- min(n, nonsilent[length(nonsilent)] * hop_length)
  samples[(start + 1L):end]
}

# voice_encoder$inference: overlapping partial windows over the mel, batched
# forward, per-utterance mean + L2 norm. rate 1.3 gives frame_step 77 (the
# Python get_frame_step), not the overlap-derived 80.
.yq_ve_embed_utterance <- function(mel, w, win_size = 160L, overlap = 0.5,
                                   rate = 1.3, min_coverage = 0.8,
                                   sample_rate = 16000L) {
  m <- as.array(mel) # [1, T, 40] host-side windowing
  n_frames <- dim(m)[2L]
  n_mels <- dim(m)[3L]
  frame_step <- if (is.null(rate)) {
    as.integer(round(win_size * (1 - overlap)))
  } else {
    as.integer(round((sample_rate / rate) / win_size))
  }
  d <- max(n_frames - win_size + frame_step, 0)
  n_partials <- d %/% frame_step
  remainder <- d %% frame_step
  if (n_partials == 0 ||
    (remainder + (win_size - frame_step)) / win_size >= min_coverage) {
    n_partials <- n_partials + 1
  }
  target_n <- win_size + frame_step * (n_partials - 1)
  if (target_n > n_frames) {
    padded <- array(0, c(1L, target_n, n_mels))
    padded[1L, seq_len(n_frames), ] <- m[1L, , ]
    m <- padded
  }
  partials <- array(0, c(n_partials, win_size, n_mels))
  for (i in seq_len(n_partials)) {
    s <- (i - 1L) * frame_step
    partials[i, , ] <- m[1L, (s + 1L):(s + win_size), ]
  }
  pe <- as.array(yq_voice_encoder(anvl::nv_array(partials, dtype = "f32"), w))
  v <- colMeans(pe) # mean over partials, then L2 normalize
  matrix(v / sqrt(sum(v^2)), nrow = 1L)
}

# S3 tokenizer on raw 16 kHz audio: log-mel frontend + optional token cap
# (max_len tokens = max_len * 4 mel frames, matching s3_tokenizer$forward).
.yq_s3_tokenize <- function(audio16, w, max_len = NULL) {
  mel <- yq_s3_log_mel_spectrogram(audio16)
  if (!is.null(max_len)) {
    tmax <- min(anvl::shape(mel)[3L], max_len * 4L)
    mel <- yunque::slice_lastdim(mel, 1L, tmax)
  }
  yq_s3tokenizer(mel, w)$tokens
}

# CAMPPlus xvector on raw 16 kHz audio: kaldi fbank with per-feature mean
# removal over frames (compute_xvector_embedding).
.yq_xvector <- function(audio16, w) {
  kf <- as.array(yq_kaldi_fbank(audio16)) # [T, 80]
  kf <- kf - matrix(colMeans(kf), nrow(kf), ncol(kf), byrow = TRUE)
  yq_campplus(anvl::nv_array(array(kf, c(1L, dim(kf))), dtype = "f32"), w)
}

#' Voice embedding for TTS conditioning (anvl)
#'
#' Torch-free port of \code{create_voice_embedding} (standard model, no
#' loudness normalization) plus \code{s3gen$embed_ref}: voice-encoder
#' speaker embedding over the full silence-trimmed reference, S3 prompt
#' tokens over the 6 s head, and the S3Gen reference dict (24 kHz prompt
#' mel, CAMPPlus xvector, aligned prompt tokens) over the 10 s head.
#'
#' @param samples Numeric reference audio.
#' @param sr Sample rate of \code{samples}.
#' @param w List of component weights: \code{ve}
#'   (\code{\link{yq_ve_load_weights}}), \code{campplus}
#'   (\code{\link{yq_campplus_load_weights}}), \code{s3tok}
#'   (\code{\link{yq_s3tokenizer_load_weights}}).
#' @param speech_cond_prompt_len T3 conditioning prompt token cap
#'   (default 150, the standard model's \code{speech_cond_prompt_len}).
#'
#' @return List: \code{ve_embedding} \code{[1, 256]},
#'   \code{cond_prompt_speech_tokens} \code{[1, <=150]} (0-based), and
#'   \code{ref_dict} with \code{prompt_token}, \code{prompt_feat}
#'   \code{[1, T, 80]}, \code{embedding} \code{[1, 192]}.
#'
#' @export
yq_voice_embedding <- function(samples, sr, w, speech_cond_prompt_len = 150L) {
  samples <- as.numeric(samples)
  s16 <- yq_resample(samples, sr, 16000)
  dec <- samples[seq_len(min(length(samples), as.integer(10 * sr)))]
  enc16 <- s16[seq_len(min(length(s16), 6L * 16000L))]

  # voice-encoder speaker embedding: full reference, silence-trimmed
  ve_mel <- yq_compute_ve_mel(.yq_trim_silence(s16))
  ve <- .yq_ve_embed_utterance(ve_mel, w$ve)

  # T3 conditioning prompt tokens (6 s cap)
  cond_tokens <- .yq_s3_tokenize(enc16, w$s3tok,
    max_len = speech_cond_prompt_len)

  # S3Gen reference dict (10 s cap)
  d24 <- yq_resample(dec, sr, 24000)
  prompt_feat <- aperm(as.array(yq_compute_mel_spectrogram(d24)),
    c(1L, 3L, 2L)) # [1, T, 80]
  d16 <- yq_resample(dec, sr, 16000)
  emb <- as.array(.yq_xvector(d16, w$campplus))
  ptok <- .yq_s3_tokenize(d16, w$s3tok)
  # keep mel and token prompts aligned: mel_len must equal 2 * token_len
  n_mel <- dim(prompt_feat)[2L]
  if (n_mel != 2L * ncol(ptok)) {
    ptok <- ptok[, seq_len(n_mel %/% 2L), drop = FALSE]
  }
  list(
    ve_embedding = ve,
    cond_prompt_speech_tokens = cond_tokens,
    ref_dict = list(prompt_token = ptok, prompt_feat = prompt_feat,
      embedding = emb)
  )
}

# drop_invalid_tokens (s3tokenizer.R): keep the span after the first SOS
# and before the first EOS, then drop remaining out-of-vocab ids.
.yq_drop_invalid_tokens <- function(tokens, vocab = 6561L) {
  vals <- as.integer(tokens)
  s <- match(vocab, vals) # SOS
  s <- if (is.na(s)) 1L else s + 1L
  e <- match(vocab + 1L, vals) # EOS
  e <- if (is.na(e)) length(vals) else e - 1L
  keep <- rep(FALSE, length(vals))
  if (s <= e) {
    keep[s:e] <- vals[s:e] < vocab
  }
  vals[keep]
}

#' Generate speech from text tokens (anvl, end to end)
#'
#' Torch-free port of the \code{generate()} chain (standard model): T3
#' CFG sampling over the conditioned Llama backbone, invalid-token
#' filtering, CFM flow to mel, HiFT vocoder to waveform, and the 20 ms
#' cosine fade-in. Text normalization and BPE tokenization stay with the
#' caller (they are pure R in \code{tokenizer.R} / \code{tts.R}).
#'
#' @param text_ids Integer vector of 0-based text token ids (no
#'   start/stop wrapping; it is added here).
#' @param voice List from \code{\link{yq_voice_embedding}}.
#' @param w List of component weights: \code{t3}, \code{cond},
#'   \code{llama}, \code{flow}, \code{hifigan} (from the respective
#'   \code{yq_*_load_weights}).
#' @param flow_noise Standard-normal noise matrix \code{[80, T]} with
#'   \code{T >= 2 * (n_prompt + n_gen)} for the CFM initial state (sliced
#'   like the reference \code{rand_noise} buffer).
#' @param exaggeration,cfg_weight,temperature,top_p,min_p
#'   ,repetition_penalty,max_new_tokens T3 sampling controls
#'   (defaults match \code{generate()}).
#' @param n_cfm_timesteps CFM Euler steps (default 10).
#' @param source_phase,source_noise Optional vocoder RNG injections (see
#'   \code{\link{yq_hifigan}}); NULL draws from the R RNG.
#' @param speech_tokens Optional pre-generated 0-based speech tokens; skips
#'   the T3 stage (used for deterministic e2e parity fixtures).
#'
#' @return List: \code{audio} (numeric, 24 kHz), \code{sample_rate},
#'   \code{speech_tokens}, \code{mel} \code{[1, 80, 2 * n_tokens]}.
#'
#' @export
yq_generate <- function(text_ids, voice, w, flow_noise,
                        exaggeration = 0.5, cfg_weight = 0.5,
                        temperature = 0.8, top_p = 1, min_p = 0.05,
                        repetition_penalty = 1.2, max_new_tokens = 1000L,
                        n_cfm_timesteps = 10L, source_phase = NULL,
                        source_noise = NULL, speech_tokens = NULL) {
  config <- .yq_t3_config()
  if (is.null(speech_tokens)) {
    if (length(text_ids) > config$max_text_tokens) {
      stop("Input text is too long: ", length(text_ids), " text tokens ",
        "exceed the T3 limit of ", config$max_text_tokens, call. = FALSE)
    }
    tt <- matrix(c(config$start_text_token, as.integer(text_ids),
      config$stop_text_token), nrow = 1L)
    prompt_emb <- .yq_t3_embed(w$t3$speech_emb, w$t3$speech_pos,
      voice$cond_prompt_speech_tokens)
    cond_emb <- yq_t3_cond_enc(
      anvl::nv_array(voice$ve_embedding, dtype = "f32"), prompt_emb,
      exaggeration, w$cond)
    speech_tokens <- yq_t3_generate(cond_emb, tt, w$t3, w$llama, config,
      max_new = max_new_tokens, temperature = temperature,
      cfg_weight = cfg_weight, top_p = top_p, min_p = min_p,
      repetition_penalty = repetition_penalty)
  }
  speech_tokens <- .yq_drop_invalid_tokens(speech_tokens)

  mel <- yq_flow_inference(speech_tokens, voice$ref_dict$prompt_token,
    voice$ref_dict$prompt_feat, voice$ref_dict$embedding, w$flow,
    flow_noise, n_timesteps = n_cfm_timesteps)

  res <- yq_hifigan(mel, w$hifigan, phase = source_phase,
    noise = source_noise)
  audio <- as.numeric(as.array(res$audio))

  # 20 ms fade-in (s3gen trim_fade): first n_trim samples zeroed, next
  # n_trim cosine-ramped
  n_trim <- 24000L %/% 50L
  fade <- c(rep(0, n_trim), (cos(seq(pi, 0, length.out = n_trim)) + 1) / 2)
  k <- seq_len(min(length(fade), length(audio)))
  audio[k] <- audio[k] * fade[k]

  list(audio = audio, sample_rate = 24000L, speech_tokens = speech_tokens,
    mel = mel)
}
