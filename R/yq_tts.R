# anvl/yunque e2e glue: the voice-embedding (reference conditioning) stage.
# Orchestrates the ported frontends + encoders exactly as the torch side's
# create_voice_embedding / compute_speaker_embedding /
# compute_xvector_embedding / s3gen$embed_ref chain does.

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
