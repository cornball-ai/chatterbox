# Text tokenizer for chatteRbox
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
    id <- vocab[[token]] + 1  # R is 1-indexed
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
    c("\u2026", ", "),  # ellipsis
    c(":", ","),
    c(" - ", ", "),
    c(";", ", "),
    c("\u2014", "-"),  # em dash
    c("\u2013", "-"),  # en dash
    c(" ,", ","),
    c("\u201c", "\""),  # left double quote
    c("\u201d", "\""),  # right double quote
    c("\u2018", "'"),   # left single quote
    c("\u2019", "'")    # right single quote
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

#' Encode text to token IDs (simple character-level fallback)
#'
#' This is a simplified encoder. For full BPE support, would need
#' to implement the merge algorithm.
#'
#' @param tokenizer Tokenizer object
#' @param text Input text
#' @return Integer vector of token IDs
tokenize_text <- function(tokenizer, text) {
  # Replace spaces with SPACE token
  text <- gsub(" ", SPECIAL_TOKENS$SPACE, text)

  # Try to find tokens greedily (longest match first)
  tokens <- character(0)
  pos <- 1
  text_len <- nchar(text)

  while (pos <= text_len) {
    found <- FALSE

    # Try decreasing lengths
    for (len in min(20, text_len - pos + 1):1) {
      substr_text <- substr(text, pos, pos + len - 1)

      if (substr_text %in% names(tokenizer$vocab)) {
        tokens <- c(tokens, substr_text)
        pos <- pos + len
        found <- TRUE
        break
      }
    }

    if (!found) {
      # Use UNK token for unknown characters
      tokens <- c(tokens, SPECIAL_TOKENS$UNK)
      pos <- pos + 1
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
decode_tokens <- function(tokenizer, ids) {
  if (inherits(ids, "torch_tensor")) {
    ids <- as.integer(ids$cpu())
  }

  tokens <- character(length(ids))
  for (i in seq_along(ids)) {
    idx <- ids[i] + 1  # Convert to R indexing
    if (idx > 0 && idx <= length(tokenizer$id_to_token)) {
      tokens[i] <- tokenizer$id_to_token[idx]
    } else {
      tokens[i] <- SPECIAL_TOKENS$UNK
    }
  }

  # Join and clean up
  text <- paste(tokens, collapse = "")
  text <- gsub(" ", "", text)  # Remove spaces from BPE
  text <- gsub(SPECIAL_TOKENS$SPACE, " ", text, fixed = TRUE)
  text <- gsub(SPECIAL_TOKENS$EOT, "", text, fixed = TRUE)
  text <- gsub(SPECIAL_TOKENS$UNK, "", text, fixed = TRUE)

  text
}
