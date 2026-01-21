# Text tokenizer for chatterbox
# Parses HuggingFace tokenizers JSON format

# Special tokens
SPECIAL_TOKENS <- list(
  SOT = "[START]",
  EOT = "[STOP]",
  UNK = "[UNK]",
  SPACE = "[SPACE]",
  PAD = "[PAD]",
  SEP = "[SEP]",
  CLS = "[CLS]",
  MASK = "[MASK]"
)

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
    id <- vocab[[token]] + 1# R is 1-indexed
    id_to_token[id] <- token
    token_to_id[token] <- vocab[[token]]
  }

  # Verify special tokens
  if (!(SPECIAL_TOKENS$SOT %in% names(vocab))) {
    warning("Missing [START] token in vocabulary")
  }
  if (!(SPECIAL_TOKENS$EOT %in% names(vocab))) {
    warning("Missing [STOP] token in vocabulary")
  }

  list(
    vocab = vocab,
    merges = merges,
    id_to_token = id_to_token,
    token_to_id = token_to_id,
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
    text <- paste0(toupper(substr(text, 1, 1)), substr(text, 2, nchar(text)))
  }

  # Remove multiple spaces
  text <- gsub("\\s+", " ", text)

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
#' Implements proper BPE (Byte Pair Encoding) using the merge list.
#' Merges are applied in priority order (first merge = highest priority).
#'
#' @param tokenizer Tokenizer object
#' @param text Input text
#' @return Integer vector of token IDs
tokenize_text <- function(
  tokenizer,
  text
) {
  # First, split text while preserving spaces as special tokens
  # Don't replace spaces yet - handle them during initial tokenization

  # Start with characters, but treat spaces as [SPACE] tokens
  chars <- strsplit(text, "") [[1]]

  # Handle characters - map spaces to [SPACE], unknowns to UNK
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
    if (is.null(best_idx)) break

    # Apply the merge
    merged_token <- paste0(tokens[best_idx], tokens[best_idx + 1])

    # Check if merged token is in vocabulary
    if (merged_token %in% names(tokenizer$vocab)) {
      # Replace the pair with merged token
      if (best_idx == 1) {
        tokens <- c(merged_token, tokens[(best_idx + 2) :length(tokens)])
      } else if (best_idx + 1 == length(tokens)) {
        tokens <- c(tokens[1:(best_idx - 1)], merged_token)
      } else {
        tokens <- c(tokens[1:(best_idx - 1)], merged_token, tokens[(best_idx + 2) :length(tokens)])
      }
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
text_to_tokens <- function(
  tokenizer,
  text,
  normalize = TRUE,
  device = "cpu"
) {
  if (normalize) {
    text <- punc_norm(text)
  }

  ids <- tokenize_text(tokenizer, text)

  torch::torch_tensor(
    matrix(ids, nrow = 1),
    dtype = torch::torch_long(),
    device = device
  )
}

#' Decode token IDs to text
#'
#' @param tokenizer Tokenizer object
#' @param ids Integer vector or tensor of token IDs
#' @return Decoded text string
#' @export
decode_tokens <- function(
  tokenizer,
  ids
) {
  if (inherits(ids, "torch_tensor")) {
    ids <- as.integer(ids$cpu())
  }

  tokens <- character(length(ids))
  for (i in seq_along(ids)) {
    idx <- ids[i] + 1# Convert to R indexing
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

