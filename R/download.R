# HuggingFace model download utilities for chatteRbox

#' Get default cache directory
#'
#' @return Path to cache directory
#' @keywords internal
get_cache_dir <- function() {
  # Check for environment variable first
  cache_dir <- Sys.getenv("CHATTERBOX_CACHE")
  if (nchar(cache_dir) > 0) {
    return(cache_dir)
  }

  # Default to ~/.cache/chatterbox
 file.path(Sys.getenv("HOME"), ".cache", "chatterbox")
}

#' Download file from HuggingFace Hub
#'
#' @param repo_id Repository ID (e.g., "ResembleAI/chatterbox")
#' @param filename Filename to download
#' @param cache_dir Cache directory (default: ~/.cache/chatterbox)
#' @param force Re-download even if file exists
#' @param timeout Download timeout in seconds (default 600)
#' @return Local path to downloaded file
#' @importFrom utils download.file
#' @export
hf_download <- function(repo_id, filename, cache_dir = NULL, force = FALSE, timeout = 600) {
  if (is.null(cache_dir)) {
    cache_dir <- get_cache_dir()
  }

  # Create cache directory structure
  repo_cache <- file.path(cache_dir, gsub("/", "--", repo_id))
  local_path <- file.path(repo_cache, filename)

  # Return existing file if present and not forcing
  if (file.exists(local_path) && !force) {
    message("Using cached: ", filename)
    return(local_path)
  }

  # Create directory
  dir.create(dirname(local_path), recursive = TRUE, showWarnings = FALSE)

  # Construct URL
  url <- sprintf("https://huggingface.co/%s/resolve/main/%s", repo_id, filename)

  # Set longer timeout for large files

  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = timeout)

  message("Downloading: ", filename)
  tryCatch({
    download.file(url, local_path, mode = "wb", quiet = FALSE)
    local_path
  }, error = function(e) {
    stop("Failed to download ", filename, " from ", repo_id, ": ", e$message)
  })
}

#' Download all chatterbox model files
#'
#' @param cache_dir Cache directory
#' @param force Re-download all files
#' @return Named list of local file paths
#' @export
download_chatterbox_models <- function(cache_dir = NULL, force = FALSE) {
  repo_id <- "ResembleAI/chatterbox"

  files <- c(
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt"
  )

  paths <- list()
  for (f in files) {
    name <- tools::file_path_sans_ext(basename(f))
    paths[[name]] <- hf_download(repo_id, f, cache_dir, force)
  }

  paths
}

#' Check if models are downloaded
#'
#' @param cache_dir Cache directory
#' @return Logical indicating if all models are present
#' @export
models_available <- function(cache_dir = NULL) {
  if (is.null(cache_dir)) {
    cache_dir <- get_cache_dir()
  }

  repo_cache <- file.path(cache_dir, "ResembleAI--chatterbox")

  files <- c(
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt"
  )

  all(file.exists(file.path(repo_cache, files)))
}
