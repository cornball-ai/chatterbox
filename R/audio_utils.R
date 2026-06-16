# Audio utilities for chatterbox
# Handles audio I/O, resampling, and mel spectrogram computation

# Detect the real container from magic bytes, ignoring the file extension.
# Voice libraries sometimes carry a .mp3 name on a real WAV (or vice versa);
# running the wrong decoder yields silent NaN garbage, not an error. Returns
# "wav", "mp3", or NA when the header is unrecognized.
.sniff_audio_format <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))
    magic <- readBin(con, "raw", n = 12L)
    if (length(magic) < 2L) {
        return(NA_character_)
    }
    if (length(magic) >= 12L &&
        identical(magic[1:4], charToRaw("RIFF")) &&
        identical(magic[9:12], charToRaw("WAVE"))) {
        return("wav")
    }
    if (length(magic) >= 3L && identical(magic[1:3], charToRaw("ID3"))) {
        return("mp3")
    }
    # MP3 frame sync: 0xFF followed by 0b111xxxxx
    if (magic[1] == as.raw(0xFF) &&
        bitwAnd(as.integer(magic[2]), 0xE0L) == 0xE0L) {
        return("mp3")
    }
    NA_character_
}

#' Read audio file
#'
#' @param path Path to audio file (WAV or MP3 format)
#' @return List with samples (numeric vector normalized to \[-1, 1\]) and sr (sample rate)
#' @export
read_audio <- function(path) {
    # Decode by actual content, not the extension (a mislabeled reference
    # otherwise runs the wrong decoder and yields NaN garbage).
    fmt <- .sniff_audio_format(path)
    if (is.na(fmt)) {
        fmt <- tolower(tools::file_ext(path))
    }

    if (fmt == "mp3") {
        # MP3 requires mpg123 system library
        wav <- tuneR::readMP3(path)
    } else {
        wav <- tuneR::readWave(path)
    }

    # Stereo downmix by channel mean (librosa.load mono= default; the
    # old behavior silently dropped the right channel). as.numeric
    # first: integer + integer overflows to NA on 32-bit samples.
    if (wav@stereo) {
        raw <- (as.numeric(wav@left) + as.numeric(wav@right)) / 2
    } else {
        raw <- wav@left
    }

    # Normalize to [-1, 1]
    if (wav@bit == 16) {
        samples <- raw / 32768
    } else if (wav@bit == 24) {
        samples <- raw / 8388608
    } else if (wav@bit == 32) {
        samples <- raw / 2147483648
    } else {
        samples <- raw / (2 ^ (wav@bit - 1))
    }

    list(samples = as.numeric(samples), sr = wav@samp.rate)
}

#' Write audio file
#'
#' @param samples Numeric vector of audio samples (normalized to \[-1, 1\])
#' @param sr Sample rate
#' @param path Output path (WAV format)
#' @export
write_audio <- function(samples, sr, path) {
    # Handle torch tensor input

    if (inherits(samples, "torch_tensor")) {
        samples <- as.numeric(samples$cpu())
    }

    # Clip to valid range
    samples <- pmax(pmin(samples, 0.99), -0.99)

    # Convert to 16-bit integer
    samples_int <- as.integer(samples * 32767)

    wav <- tuneR::Wave(left = samples_int, samp.rate = as.integer(sr), bit = 16)

    # extensible = FALSE writes plain WAVE_FORMAT_PCM. The default
    # (extensible = TRUE) writes WAVE_FORMAT_EXTENSIBLE and stamps a mono
    # file with channel mask SPEAKER_FRONT_LEFT, so players route it to the
    # left speaker only; plain PCM mono plays on both.
    tuneR::writeWave(wav, path, extensible = FALSE)
}

#' Resample audio
#'
#' @param samples Numeric vector of audio samples
#' @param from_sr Source sample rate
#' @param to_sr Target sample rate
#' @return Resampled audio samples
#' @export
resample_audio <- function(samples, from_sr, to_sr) {
    if (from_sr == to_sr) {
        return(samples)
    }

    # Windowed-sinc resampling (torchaudio-equivalent); the previous
    # linear interpolation aliased content above the target Nyquist
    # into the reference conditioning features
    sinc_resample(samples, from_sr, to_sr)
}

#' Trim leading and trailing silence
#'
#' Port of librosa.effects.trim: frame power is compared against the
#' loudest frame; frames more than \code{top_db} below it are silence.
#'
#' @param samples Numeric vector of audio samples
#' @param top_db Threshold in dB below the peak frame power (default 20)
#' @param frame_length Analysis frame length (default 2048)
#' @param hop_length Hop between frames (default 512)
#' @return Trimmed audio samples
#' @noRd
trim_silence <- function(samples, top_db = 20, frame_length = 2048L,
                         hop_length = 512L) {
    n <- length(samples)
    if (n == 0) {
        return(samples)
    }

    # Centered framing with zero padding (librosa.feature.rms defaults)
    pad <- frame_length %/% 2L
    padded <- c(rep(0, pad), samples, rep(0, pad))
    n_frames <- 1L + (length(padded) - frame_length) %/% hop_length
    power <- vapply(seq_len(n_frames), function(i) {
        s <- (i - 1L) * hop_length
        mean(padded[(s + 1L):(s + frame_length)] ^ 2)
    }, numeric(1))

    ref <- max(power)
    if (is.na(ref)) {
        stop("trim_silence: NaN in reference-audio power - the voice encoding ",
             "is corrupt (a CUDA allocator race can cause this); retry, or ",
             "report if it persists", call. = FALSE)
    }
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

#' Create mel filterbank
#'
#' @param sr Sample rate
#' @param n_fft FFT size
#' @param n_mels Number of mel bins
#' @param fmin Minimum frequency
#' @param fmax Maximum frequency
#' @param norm Character. Normalization type. Default "slaney".
#' @param htk Logical. Use HTK formula. Default FALSE.
#' @return Mel filterbank matrix (n_mels x (n_fft/2 + 1))
create_mel_filterbank <- function(sr, n_fft, n_mels, fmin = 0, fmax = NULL,
                                  norm = "slaney", htk = FALSE) {
    if (is.null(fmax)) {
        fmax <- sr / 2
    }

    if (htk) {
        # HTK formula (not used by librosa default)
        hz_to_mel <- function(hz)
        {
            2595 * log10(1 + hz / 700)
        }
        mel_to_hz <- function(mel)
        {
            700 * (10 ^ (mel / 2595) - 1)
        }
    } else {
        # Slaney/librosa formula (default)
        # Linear below 1000 Hz, log above
        f_sp <- 200.0 / 3 # 66.67 Hz per mel below 1000 Hz
        min_log_hz <- 1000.0
        min_log_mel <- (min_log_hz - 0) / f_sp # 15.0
        logstep <- log(6.4) / 27.0 # step size for log region

        hz_to_mel <- function(hz)
        {
            ifelse(hz < min_log_hz, hz / f_sp,
                   min_log_mel + log(hz / min_log_hz) / logstep)
        }
        mel_to_hz <- function(mel)
        {
            ifelse(mel < min_log_mel,
                   mel * f_sp,
                   min_log_hz * exp(logstep * (mel - min_log_mel)))
        }
    }

    # Create mel points
    mel_min <- hz_to_mel(fmin)
    mel_max <- hz_to_mel(fmax)
    mel_points <- seq(mel_min, mel_max, length.out = n_mels + 2)
    hz_points <- mel_to_hz(mel_points)

    # Convert to FFT bin numbers
    n_fft_bins <- n_fft %/% 2 + 1
    fft_freqs <- seq(0, sr / 2, length.out = n_fft_bins)

    # Create filterbank
    filterbank <- matrix(0, nrow = n_mels, ncol = n_fft_bins)

    for (i in seq_len(n_mels)) {
        left <- hz_points[i]
        center <- hz_points[i + 1]
        right <- hz_points[i + 2]

        # Rising edge
        rising <- (fft_freqs - left) / (center - left)
        rising[fft_freqs < left] <- 0
        rising[fft_freqs > center] <- 0

        # Falling edge
        falling <- (right - fft_freqs) / (right - center)
        falling[fft_freqs < center] <- 0
        falling[fft_freqs > right] <- 0

        filterbank[i,] <- pmax(rising, 0) + pmax(falling, 0)
        filterbank[i, fft_freqs >= center] <- pmax(falling[fft_freqs >= center], 0)
        filterbank[i, fft_freqs < center] <- pmax(rising[fft_freqs < center], 0)
    }

    # Apply Slaney normalization (divide by bandwidth in Hz)
    # This matches librosa's default norm="slaney"
    if (norm == "slaney") {
        # enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
        # Bandwidth is the difference between upper and lower Hz for each filter
        enorm <- 2.0 / (hz_points[3:(n_mels + 2)] - hz_points[1:n_mels])
        filterbank <- filterbank * enorm
    }

    filterbank
}

# Cache for mel filterbanks and hann windows
.mel_cache <- new.env(parent = emptyenv())

#' Compute mel spectrogram (S3Gen compatible)
#'
#' @param y Audio samples as torch tensor or numeric vector
#' @param n_fft FFT size (default 1920 for 24kHz)
#' @param n_mels Number of mel bins (default 80)
#' @param sr Sample rate (default 24000)
#' @param hop_size Hop size (default 480)
#' @param win_size Window size (default 1920)
#' @param fmin Minimum frequency (default 0)
#' @param fmax Maximum frequency (default 8000)
#' @param center Whether to center frames (default FALSE)
#' @return Mel spectrogram tensor (batch, n_mels, time)
#' @export
compute_mel_spectrogram <- function(y, n_fft = 1920, n_mels = 80, sr = 24000,
                                    hop_size = 480, win_size = 1920,
                                    fmin = 0, fmax = 8000, center = FALSE) {
    # Convert to torch tensor if needed
    if (!inherits(y, "torch_tensor")) {
        y <- torch::torch_tensor(y, dtype = torch::torch_float32())
    }

    # Add batch dimension if needed
    if (y$dim() == 1) {
        y <- y$unsqueeze(1)
    }

    device <- y$device

    # Get or create mel filterbank (key includes all parameters that affect shape)
    mel_cache_key <- paste(sr, n_fft, n_mels, fmin, fmax, device$type,
                           sep = "_")
    if (is.null(.mel_cache[[mel_cache_key]])) {
        mel_fb <- create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        .mel_cache[[mel_cache_key]] <- torch::torch_tensor(mel_fb, dtype = torch::torch_float32())$to(device = device)
    }
    mel_basis <- .mel_cache[[mel_cache_key]]

    # Get or create Hann window (key includes win_size)
    win_key <- paste("hann", win_size, device$type, sep = "_")
    if (is.null(.mel_cache[[win_key]])) {
        .mel_cache[[win_key]] <- torch::torch_hann_window(win_size)$to(device = device)
    }
    hann_window <- .mel_cache[[win_key]]

    # Pad audio (reflect padding)
    pad_amount <- as.integer((n_fft - hop_size) / 2)
    y <- y$unsqueeze(2) # Add channel dim for padding
    y <- torch::nnf_pad(y, c(pad_amount, pad_amount), mode = "reflect")
    y <- y$squeeze(2)

    # Compute STFT
    spec <- torch::torch_stft(
                              y,
                              n_fft = n_fft,
                              hop_length = hop_size,
                              win_length = win_size,
                              window = hann_window,
                              center = center,
                              pad_mode = "reflect",
                              normalized = FALSE,
                              onesided = TRUE,
                              return_complex = TRUE
    )

    # Convert to magnitude
    spec <- torch::torch_view_as_real(spec)
    spec <- torch::torch_sqrt(spec$pow(2)$sum(-1) + 1e-9)

    # Apply mel filterbank
    spec <- torch::torch_matmul(mel_basis, spec)

    # Log compression
    spec <- torch::torch_log(torch::torch_clamp(spec, min = 1e-5))

    spec
}

#' Compute mel spectrogram for voice encoder (40 bins, 16kHz)
#'
#' @param y Audio samples
#' @param sr Sample rate (should be 16000)
#' @return Mel spectrogram (batch, time, 40)
#' @export
compute_mel_spectrogram_ve <- function(y, sr = 16000) {
    # Voice encoder uses different params
    spec <- compute_mel_spectrogram(y, n_fft = 400, n_mels = 40, sr = sr,
                                    hop_size = 160, win_size = 400, fmin = 0,
                                    fmax = 8000, center = TRUE)

    # Transpose to (batch, time, mels) for LSTM
    spec$transpose(2, 3)
}
