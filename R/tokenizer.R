# Text tokenizer for chatterbox
# Parses HuggingFace tokenizers JSON format

# Special tokens
SPECIAL_TOKENS <- list(SOT = "[START]", EOT = "[STOP]", UNK = "[UNK]",
                       SPACE = "[SPACE]", PAD = "[PAD]", SEP = "[SEP]",
                       CLS = "[CLS]", MASK = "[MASK]")

#' Load tokenizer from JSON file
#'
#' @param vocab_path Path to tokenizer.json
#' @return Tokenizer object (list)
#' @export
load_bpe_tokenizer <- function(vocab_path) {
    load_tokenizer(vocab_path)
}

#' Load tokenizer from JSON file (internal)
#'
#' @param vocab_path Path to tokenizer.json
#' @return Tokenizer object (list)
load_tokenizer <- function(vocab_path) {
    json_data <- jsonlite::fromJSON(vocab_path, simplifyVector = FALSE)

    # Extract vocabulary
    vocab <- list()
    if (!is.null(json_data$model$vocab)) {
        vocab <- json_data$model$vocab
    }

    # Extract merges for BPE
    merges <- character(0)
    if (!is.null(json_data$model$merges)) {
        merges <- unlist(json_data$model$merges)
    }

    # Build reverse vocab (id -> token)
    id_to_token <- character(length(vocab))
    token_to_id <- integer(length(vocab))
    names(token_to_id) <- names(vocab)

    for (token in names(vocab)) {
        id <- vocab[[token]] + 1 # R is 1-indexed
        id_to_token[id] <- token
        token_to_id[token] <- vocab[[token]]
    }

    # Verify special tokens (Python asserts; a tokenizer without them is unusable)
    if (!(SPECIAL_TOKENS$SOT %in% names(vocab))) {
        stop("Missing [START] token in vocabulary: ", vocab_path)
    }
    if (!(SPECIAL_TOKENS$EOT %in% names(vocab))) {
        stop("Missing [STOP] token in vocabulary: ", vocab_path)
    }

    # Added tokens ([SPACE], [laughter], [sigh], ...) are extracted from
    # text before BPE runs, exactly like HF tokenizers' AddedToken pass.
    # Sort longest-first so e.g. [PLACEHOLDER55] wins over any prefix.
    added_tokens <- character(0)
    if (!is.null(json_data$added_tokens)) {
        contents <- vapply(json_data$added_tokens, function(t) t$content,
                           character(1))
        ids <- vapply(json_data$added_tokens, function(t) as.integer(t$id), integer(1))
        ord <- order(nchar(contents), decreasing = TRUE)
        added_tokens <- ids[ord]
        names(added_tokens) <- contents[ord]
    }

    list(
         vocab = vocab,
         merges = merges,
         id_to_token = id_to_token,
         token_to_id = token_to_id,
         added_tokens = added_tokens,
         vocab_size = length(vocab)
    )
}

#' Normalize punctuation for TTS
#'
#' @param text Input text
#' @return Normalized text
punc_norm <- function(text) {
    if (nchar(text) == 0) {
        return("You need to add some text for me to talk.")
    }

    # Capitalize first letter
    if (grepl("^[a-z]", text)) {
        text <- paste0(toupper(substr(text, 1, 1)),
                       substr(text, 2, nchar(text)))
    }

    # Remove multiple spaces (Python: " ".join(text.split()) — also trims both ends)
    text <- trimws(gsub("\\s+", " ", text))

    # Replace uncommon punctuation
    replacements <- list(
                         c("...", ", "),
                         c("\u2026", ", "), # ellipsis
                         c(":", ","),
                         c(" - ", ", "),
                         c(";", ", "),
                         c("\u2014", "-"), # em dash
                         c("\u2013", "-"), # en dash
                         c(" ,", ","),
                         c("\u201c", "\""), # left double quote
                         c("\u201d", "\""), # right double quote
                         c("\u2018", "'"), # left single quote
                         c("\u2019", "'") # right single quote
    )

    for (r in replacements) {
        text <- gsub(r[1], r[2], text, fixed = TRUE)
    }

    # Trim trailing spaces
    text <- trimws(text, "right")

    # Add period if no ending punctuation
    sentence_enders <- c(".", "!", "?", "-", ",")
    last_char <- substr(text, nchar(text), nchar(text))
    if (!(last_char %in% sentence_enders)) {
        text <- paste0(text, ".")
    }

    text
}

#' Encode text to token IDs using BPE
#'
#' Mirrors the HF tokenizers pipeline used by the Python reference:
#' added tokens ([SPACE], [laughter], [sigh], ...) are extracted first
#' and map directly to their ids; BPE merges run on the remaining text,
#' in priority order (first merge = highest priority).
#'
#' @param tokenizer Tokenizer object
#' @param text Input text
#' @return Integer vector of token IDs
tokenize_text <- function(tokenizer, text) {
    ids <- integer(0)
    for (seg in split_added_tokens(text, tokenizer$added_tokens)) {
        if (!is.na(seg$id)) {
            ids <- c(ids, seg$id)
        } else {
            ids <- c(ids, bpe_encode(tokenizer, seg$text))
        }
    }
    ids
}

#' Split text on added-token occurrences
#'
#' HF tokenizers extract added tokens before pre-tokenization/BPE, so
#' \code{[laughter]} is one token, never spelled out letter by letter.
#' \code{added_tokens} is sorted longest-first at load time so prefix
#' collisions resolve the same way (regex alternation is first-match).
#'
#' @param text Input text
#' @param added_tokens Named integer vector (content -> id)
#' @return List of segments: \code{list(text =, id =)}, id NA for plain text
#' @noRd
split_added_tokens <- function(text, added_tokens) {
    if (length(added_tokens) == 0 || !nzchar(text)) {
        return(list(list(text = text, id = NA_integer_)))
    }
    escaped <- gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", names(added_tokens))
    m <- gregexpr(paste(escaped, collapse = "|"), text)[[1]]
    if (m[1] == -1L) {
        return(list(list(text = text, id = NA_integer_)))
    }
    lens <- attr(m, "match.length")
    segments <- list()
    pos <- 1L
    for (i in seq_along(m)) {
        if (m[i] > pos) {
            segments[[length(segments) + 1L]] <- list(
                text = substr(text, pos, m[i] - 1L), id = NA_integer_
            )
        }
        content <- substr(text, m[i], m[i] + lens[i] - 1L)
        segments[[length(segments) + 1L]] <- list(text = content,
            id = unname(added_tokens[[content]]))
        pos <- m[i] + lens[i]
    }
    if (pos <= nchar(text)) {
        segments[[length(segments) + 1L]] <- list(
            text = substr(text, pos, nchar(text)), id = NA_integer_
        )
    }
    segments
}

#' BPE-encode a plain text segment (no added tokens)
#'
#' @param tokenizer Tokenizer object
#' @param text Text segment
#' @return Integer vector of token IDs
#' @noRd
bpe_encode <- function(tokenizer, text) {
    # Non-space whitespace is discarded by HF's Whitespace pre-tokenizer
    # but still separates words: BPE must not merge across it
    # ("a\tb" is [a, b], never the merged token "ab")
    pieces <- strsplit(text, "(?:(?! )\\s)+", perl = TRUE)[[1]]
    if (length(pieces) > 1) {
        return(unlist(lapply(pieces, function(p) bpe_encode(tokenizer, p))))
    }

    chars <- strsplit(text, "")[[1]]

    # Map spaces to [SPACE]; unknown chars to [UNK]
    tokens <- character(length(chars))
    for (i in seq_along(chars)) {
        if (chars[i] == " ") {
            tokens[i] <- SPECIAL_TOKENS$SPACE
        } else if (chars[i] %in% names(tokenizer$vocab)) {
            tokens[i] <- chars[i]
        } else {
            tokens[i] <- SPECIAL_TOKENS$UNK
        }
    }

    # Build merge priority map (lower index = higher priority)
    merge_priority <- seq_along(tokenizer$merges)
    names(merge_priority) <- tokenizer$merges

    # Apply BPE merges iteratively
    while (length(tokens) > 1) {
        # Find the highest priority merge that can be applied
        best_idx <- NULL
        best_priority <- Inf

        for (i in seq_len(length(tokens) - 1)) {
            pair <- paste(tokens[i], tokens[i + 1])
            if (pair %in% names(merge_priority)) {
                if (merge_priority[[pair]] < best_priority) {
                    best_priority <- merge_priority[[pair]]
                    best_idx <- i
                }
            }
        }

        # No more merges possible
        if (is.null(best_idx)) {
            break
        }

        # Apply the merge
        merged_token <- paste0(tokens[best_idx], tokens[best_idx + 1])

        # Check if merged token is in vocabulary
        if (merged_token %in% names(tokenizer$vocab)) {
            if (best_idx > 1) {
                left <- tokens[1:(best_idx - 1)]
            } else {
                left <- character(0)
            }
            right <- if (best_idx + 1 < length(tokens)) {
                tokens[(best_idx + 2):length(tokens)]
            } else {
                character(0)
            }
            tokens <- c(left, merged_token, right)
        } else {
            # Merged token not in vocab, skip this merge
            break
        }
    }

    # Convert tokens to IDs
    ids <- integer(length(tokens))
    for (i in seq_along(tokens)) {
        if (tokens[i] %in% names(tokenizer$token_to_id)) {
            ids[i] <- tokenizer$token_to_id[[tokens[i]]]
        } else {
            ids[i] <- tokenizer$token_to_id[[SPECIAL_TOKENS$UNK]]
        }
    }

    ids
}

#' Convert text to token tensor
#'
#' @param tokenizer Tokenizer object
#' @param text Input text
#' @param normalize Whether to apply punctuation normalization
#' @param device Target device
#' @return Token tensor (1, seq_len)
#' @export
text_to_tokens <- function(tokenizer, text, normalize = TRUE, device = "cpu") {
    if (normalize) {
        text <- punc_norm(text)
    }

    ids <- tokenize_text(tokenizer, text)

    torch::torch_tensor(matrix(ids, nrow = 1), dtype = torch::torch_long(),
                        device = device)
}

# ============================================================================
# GPT-2 BPE Tokenizer (for Turbo model)
# ============================================================================

#' Load GPT-2 BPE tokenizer
#'
#' Loads vocab.json, merges.txt, and optionally added_tokens.json
#' for GPT-2 byte-level BPE tokenization.
#'
#' @param vocab_path Path to vocab.json
#' @param merges_path Path to merges.txt
#' @param added_tokens_path Path to added_tokens.json (optional)
#' @return Tokenizer object (list)
#' @export
load_gpt2_tokenizer <- function(vocab_path, merges_path,
                                added_tokens_path = NULL) {
    # Load vocabulary (token string -> id)
    vocab <- jsonlite::fromJSON(vocab_path, simplifyVector = TRUE)

    # Load added tokens if provided. As well as putting them in the vocab
    # (id lookup / decoding), keep a content->id map sorted longest-first
    # so tokenize_text_gpt2() can split them out before BPE - otherwise the
    # GPT-2 pre-tokenizer shatters "[sigh]" into "[", "sigh", "]" and the
    # paralinguistic tag id is never emitted.
    added_tokens <- integer(0)
    if (!is.null(added_tokens_path) && file.exists(added_tokens_path)) {
        added <- jsonlite::fromJSON(added_tokens_path, simplifyVector = TRUE)
        for (tok in names(added)) {
            vocab[[tok]] <- added[[tok]]
        }
        contents <- names(added)
        ids <- as.integer(added)
        ord <- order(nchar(contents), decreasing = TRUE)
        added_tokens <- ids[ord]
        names(added_tokens) <- contents[ord]
    }

    # Build byte-to-unicode mapping (GPT-2 specific)
    byte_encoder <- .gpt2_bytes_to_unicode()
    byte_decoder <- names(byte_encoder)
    names(byte_decoder) <- byte_encoder

    # Load merges
    merge_lines <- readLines(merges_path, warn = FALSE)
    # Skip first line (header "#version: ...")
    if (length(merge_lines) > 0 && startsWith(merge_lines[1], "#")) {
        merge_lines <- merge_lines[-1]
    }
    # Remove empty lines
    merge_lines <- merge_lines[nzchar(merge_lines)]

    # Build merge rank map (pair -> priority)
    bpe_ranks <- list()
    for (i in seq_along(merge_lines)) {
        bpe_ranks[[merge_lines[i]]] <- i
    }

    # Build reverse vocab (id -> token)
    id_to_token <- character(length(vocab))
    for (token in names(vocab)) {
        id <- vocab[[token]] + 1L # R is 1-indexed
        if (id > 0 && id <= length(vocab)) {
            id_to_token[id] <- token
        }
    }

    list(vocab = vocab, bpe_ranks = bpe_ranks, byte_encoder = byte_encoder,
         byte_decoder = byte_decoder, id_to_token = id_to_token,
         added_tokens = added_tokens,
         vocab_size = length(vocab), type = "gpt2")
}

#' GPT-2 bytes-to-unicode mapping
#' @return Named character vector (byte value as character -> unicode char)
#' @keywords internal
.gpt2_bytes_to_unicode <- function()
{
    # Printable bytes that map to themselves
    bs <- c(
            33:126, # '!' to '~'
            161:172, # inverted exclamation to not sign
            174:255 # registered sign to y-diaeresis
    )
    cs <- bs

    # Non-printable bytes get mapped to 256+ range
    n <- 0L
    for (b in 0:255) {
        if (!(b %in% bs)) {
            bs <- c(bs, b)
            cs <- c(cs, 256L + n)
            n <- n + 1L
        }
    }

    result <- intToUtf8(cs, multiple = TRUE)
    names(result) <- as.character(bs)
    result
}

#' Tokenize text using GPT-2 BPE
#'
#' @param tokenizer GPT-2 tokenizer from load_gpt2_tokenizer()
#' @param text Input text
#' @return Integer vector of token IDs
#' @keywords internal
tokenize_text_gpt2 <- function(tokenizer, text) {
    # Split out added tokens (paralinguistic/emotion tags like [sigh],
    # [laugh], [whispering]) first so each maps to its single id; BPE only
    # runs on the text in between.
    added_tokens <- tokenizer$added_tokens
    if (is.null(added_tokens)) {
        added_tokens <- integer(0)
    }
    all_ids <- integer(0)
    for (seg in split_added_tokens(text, added_tokens)) {
        if (!is.na(seg$id)) {
            all_ids <- c(all_ids, seg$id)
        } else {
            all_ids <- c(all_ids, .gpt2_bpe_encode(tokenizer, seg$text))
        }
    }
    all_ids
}

#' GPT-2 byte-level BPE for a plain text segment (no added-token handling)
#' @param tokenizer GPT-2 tokenizer
#' @param text Plain text segment
#' @return Integer vector of token IDs
#' @keywords internal
.gpt2_bpe_encode <- function(tokenizer, text) {
    byte_encoder <- tokenizer$byte_encoder
    bpe_ranks <- tokenizer$bpe_ranks
    vocab <- tokenizer$vocab

    # GPT-2 pre-tokenization regex pattern
    # Matches: contractions, words with optional leading space, numbers, punctuation, whitespace
    # \p{L}/\p{N} (Unicode letter/number), like GPT-2's regex; the POSIX
    # classes split non-ASCII letters differently
    pat <- "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

    # Find all matches
    matches <- gregexpr(pat, text, perl = TRUE)
    tokens_raw <- regmatches(text, matches)[[1]]

    all_ids <- integer(0)

    for (token_str in tokens_raw) {
        # Convert bytes to unicode representation
        raw_bytes <- charToRaw(token_str)
        bpe_token <- paste0(byte_encoder[as.character(as.integer(raw_bytes))],
                            collapse = "")

        # Apply BPE
        bpe_result <- .apply_bpe(bpe_token, bpe_ranks)

        # Look up each BPE piece in vocabulary
        for (piece in bpe_result) {
            if (piece %in% names(vocab)) {
                all_ids <- c(all_ids, vocab[[piece]])
            }
        }
    }

    all_ids
}

#' Apply BPE merges to a token
#' @param token Character string (already byte-encoded)
#' @param bpe_ranks Merge rank map
#' @return Character vector of BPE pieces
#' @keywords internal
.apply_bpe <- function(token, bpe_ranks) {
    # Split into individual characters
    word <- strsplit(token, "")[[1]]

    if (length(word) <= 1) {
        return(word)
    }

    while (length(word) > 1) {
        # Find the pair with lowest rank
        best_pair <- NULL
        best_rank <- Inf

        for (i in seq_len(length(word) - 1)) {
            pair_key <- paste(word[i], word[i + 1])
            rank <- bpe_ranks[[pair_key]]
            if (!is.null(rank) && rank < best_rank) {
                best_rank <- rank
                best_pair <- pair_key
            }
        }

        if (is.null(best_pair)) {
            break
        }

        # Merge the best pair
        parts <- strsplit(best_pair, " ")[[1]]
        first <- parts[1]
        second <- parts[2]

        new_word <- character(0)
        i <- 1L
        while (i <= length(word)) {
            if (i < length(word) && word[i] == first && word[i + 1] == second) {
                new_word <- c(new_word, paste0(first, second))
                i <- i + 2L
            } else {
                new_word <- c(new_word, word[i])
                i <- i + 1L
            }
        }

        word <- new_word
    }

    word
}

#' Decode token IDs to text
#'
#' @param tokenizer Tokenizer object
#' @param ids Integer vector or tensor of token IDs
#' @return Decoded text string
#' @export
decode_tokens <- function(tokenizer, ids) {
    if (inherits(ids, "torch_tensor")) {
        ids <- as.integer(ids$cpu())
    }

    tokens <- character(length(ids))
    for (i in seq_along(ids)) {
        idx <- ids[i] + 1 # Convert to R indexing
        if (idx > 0 && idx <= length(tokenizer$id_to_token)) {
            tokens[i] <- tokenizer$id_to_token[idx]
        } else {
            tokens[i] <- SPECIAL_TOKENS$UNK
        }
    }

    # Join and clean up
    text <- paste(tokens, collapse = "")
    text <- gsub(" ", "", text) # Remove spaces from BPE
    text <- gsub(SPECIAL_TOKENS$SPACE, " ", text, fixed = TRUE)
    text <- gsub(SPECIAL_TOKENS$EOT, "", text, fixed = TRUE)
    text <- gsub(SPECIAL_TOKENS$UNK, "", text, fixed = TRUE)

    text
}
