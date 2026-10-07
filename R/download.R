# Model Download Utilities
#
# Download Chatterbox models from HuggingFace using hfhub.
# Requires explicit download with user consent (no auto-download).

CHATTERBOX_REPO <- "ResembleAI/chatterbox"

# conds.pt (Python's builtin default voice) is intentionally absent:
# it is a nested torch pickle that R torch cannot read, and the R API
# requires a reference voice. Saves a ~105 MB download.
CHATTERBOX_FILES <- c("ve.safetensors", "t3_cfg.safetensors",
                      "s3gen.safetensors", "tokenizer.json")

# Approximate total model size in MB
.model_size_mb <- 2000

CHATTERBOX_TURBO_REPO <- "ResembleAI/chatterbox-turbo"

CHATTERBOX_TURBO_FILES <- c("t3_turbo_v1.safetensors",
                            "s3gen_meanflow.safetensors", "s3gen.safetensors",
                            "ve.safetensors", "vocab.json", "merges.txt",
                            "added_tokens.json", "special_tokens_map.json",
                            "tokenizer_config.json")

# Approximate turbo model size in MB
.turbo_model_size_mb <- 3800

# The revision every hub call in this package resolves against.
#
# NULL means hfhub's default, "main" -- a BRANCH, which it resolves through
# `refs/main` in the cache and, failing that, over the network. Passing an
# exact 40-hex commit takes hfhub's fast path instead: it goes straight to
# `snapshots/<revision>/<file>` and never consults `refs/`.
#
# That difference is what lets chatterbox run against a cache holding ONLY the
# snapshot -- a read-only bind mount of one revision, with no `refs/` and no
# network. It is also the stronger guarantee generally: a branch moves, so a
# deployment pinned to yesterday's weights would silently start resolving
# today's.
#
# A branch name is REFUSED rather than passed through. Accepting one would
# hand hfhub a value it resolves the slow way, which is the behaviour this
# argument exists to avoid, and the caller would have no way to tell.
.chatterbox_rev <- function(revision)
{
    if (is.null(revision)) {
        return(list())
    }
    if (!is.character(revision) || length(revision) != 1L || is.na(revision) ||
        !grepl("^[0-9a-f]{40}$", revision)) {
        stop("revision must be a single 40-character hex commit, not a branch ",
             "name: a branch resolves through refs/ and defeats the point of ",
             "pinning one", call. = FALSE)
    }
    list(revision = revision)
}

#' Check if Models are Downloaded
#'
#' @param revision Optional exact 40-hex commit to resolve against. With one,
#'   this answers about that snapshot and needs neither a \code{refs/} entry
#'   nor the network.
#' @return TRUE if all model files exist locally
#' @export
#' @examples
#' models_available()
models_available <- function(revision = NULL)
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        return(FALSE)
    }

    rev <- .chatterbox_rev(revision)
    tryCatch({
        for (f in CHATTERBOX_FILES) {
            do.call(hfhub::hub_download,
                    c(list(CHATTERBOX_REPO, f, local_files_only = TRUE), rev))
        }
        TRUE
    }, error = function(e) FALSE)
}

#' Download Chatterbox Models from HuggingFace
#'
#' Download all Chatterbox model files from HuggingFace.
#' In interactive sessions, asks for user consent before downloading.
#'
#' @param force Re-download even if files exist
#' @param revision Optional exact 40-hex commit. Every file is fetched at that
#'   one revision, so the resulting cache entry is a single self-contained
#'   snapshot directory.
#' @return Named list of local file paths (invisibly)
#' @export
#' @examples
#' \dontrun{
#' # Download models (~2GB)
#' download_chatterbox_models()
#' }
download_chatterbox_models <- function(force = FALSE, revision = NULL) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install it from CRAN before downloading models.")
    }
    rev <- .chatterbox_rev(revision)

    # Check if already downloaded
    if (!force && models_available(revision = revision)) {
        message("Chatterbox models are already downloaded.")
        return(invisible(get_model_paths(revision = revision)))
    }

    # Ask for consent (required for CRAN compliance)
    # Skip prompt if chatterbox.consent option is set
    if (isTRUE(getOption("chatterbox.consent"))) {
        # Consent already given programmatically
    } else if (interactive()) {
        ans <- utils::askYesNo(
                               paste0("Download Chatterbox models (~", .model_size_mb,
                                      " MB) from HuggingFace?"),
                               default = TRUE
        )
        if (!isTRUE(ans)) {
            stop("Download cancelled.", call. = FALSE)
        }
    } else {
        stop(
             "Cannot download models in non-interactive mode without consent. ",
             "Run download_chatterbox_models() interactively first, ",
             "or set options(chatterbox.consent = TRUE) to allow downloads.",
             call. = FALSE
        )
    }

    message("Downloading Chatterbox models from HuggingFace (", CHATTERBOX_REPO, ")...")

    paths <- list()
    for (f in CHATTERBOX_FILES) {
        message("  ", f, "...")
        tryCatch({
            path <- do.call(hfhub::hub_download,
                            c(list(CHATTERBOX_REPO, f, force_download = force),
                              rev))
            name <- tools::file_path_sans_ext(basename(f))
            paths[[name]] <- path
        }, error = function(e) {
            warning("Failed to download ", f, ": ", e$message)
        })
    }

    if (length(paths) < length(CHATTERBOX_FILES)) {
        stop("Failed to download all model files")
    }

    message("Models downloaded successfully.")
    invisible(paths)
}

#' Get Paths to Downloaded Model Files
#'
#' @param revision Optional exact 40-hex commit to resolve against.
#' @return Named list of local file paths
#' @keywords internal
get_model_paths <- function(revision = NULL)
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install it from CRAN before downloading models.")
    }
    rev <- .chatterbox_rev(revision)

    paths <- list()
    for (f in CHATTERBOX_FILES) {
        name <- tools::file_path_sans_ext(basename(f))
        tryCatch({
            paths[[name]] <- do.call(hfhub::hub_download,
                                     c(list(CHATTERBOX_REPO, f,
                                            local_files_only = TRUE), rev))
        }, error = function(e) {
            stop(
                 "Model file '", f, "' not found. ",
                 "Run download_chatterbox_models() first.",
                 call. = FALSE
            )
        })
    }

    paths
}

#' Check if Turbo Models are Downloaded
#'
#' @param revision Optional exact 40-hex commit to resolve against. With one,
#'   this answers about that snapshot and needs neither a \code{refs/} entry
#'   nor the network.
#' @return TRUE if all turbo model files exist locally
#' @export
#' @examples
#' turbo_models_available()
turbo_models_available <- function(revision = NULL)
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        return(FALSE)
    }

    rev <- .chatterbox_rev(revision)
    tryCatch({
        for (f in CHATTERBOX_TURBO_FILES) {
            do.call(hfhub::hub_download,
                    c(list(CHATTERBOX_TURBO_REPO, f,
                           local_files_only = TRUE), rev))
        }
        TRUE
    }, error = function(e) FALSE)
}

#' Download Chatterbox Turbo Models from HuggingFace
#'
#' Download all Chatterbox Turbo model files from HuggingFace.
#' The turbo model uses a GPT-2 backbone and MeanFlow decoder for faster inference.
#'
#' @param force Re-download even if files exist
#' @param revision Optional exact 40-hex commit. Every file is fetched at that
#'   one revision, so the resulting cache entry is a single self-contained
#'   snapshot directory.
#' @return Named list of local file paths (invisibly)
#' @export
#' @examples
#' \dontrun{
#' download_chatterbox_turbo_models()
#' }
download_chatterbox_turbo_models <- function(force = FALSE, revision = NULL) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install it from CRAN before downloading models.")
    }
    rev <- .chatterbox_rev(revision)

    if (!force && turbo_models_available(revision = revision)) {
        message("Chatterbox Turbo models are already downloaded.")
        return(invisible(get_turbo_model_paths(revision = revision)))
    }

    if (isTRUE(getOption("chatterbox.consent"))) {
        # Consent already given
    } else if (interactive()) {
        ans <- utils::askYesNo(
                               paste0("Download Chatterbox Turbo models (~",
                                      .turbo_model_size_mb, " MB) from HuggingFace?"),
                               default = TRUE
        )
        if (!isTRUE(ans)) {
            stop("Download cancelled.", call. = FALSE)
        }
    } else {
        stop(
             "Cannot download models in non-interactive mode without consent. ",
             "Run download_chatterbox_turbo_models() interactively first, ",
             "or set options(chatterbox.consent = TRUE) to allow downloads.",
             call. = FALSE
        )
    }

    message("Downloading Chatterbox Turbo models from HuggingFace (",
            CHATTERBOX_TURBO_REPO, ")...")

    paths <- list()
    for (f in CHATTERBOX_TURBO_FILES) {
        message("  ", f, "...")
        tryCatch({
            path <- do.call(hfhub::hub_download,
                            c(list(CHATTERBOX_TURBO_REPO, f,
                                   force_download = force), rev))
            name <- tools::file_path_sans_ext(basename(f))
            paths[[name]] <- path
        }, error = function(e) {
            warning("Failed to download ", f, ": ", e$message)
        })
    }

    if (length(paths) < length(CHATTERBOX_TURBO_FILES)) {
        stop("Failed to download all turbo model files")
    }

    message("Turbo models downloaded successfully.")
    invisible(paths)
}

#' Get Paths to Downloaded Turbo Model Files
#'
#' @param revision Optional exact 40-hex commit to resolve against.
#' @return Named list of local file paths
#' @keywords internal
get_turbo_model_paths <- function(revision = NULL)
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install it from CRAN before downloading models.")
    }
    rev <- .chatterbox_rev(revision)

    paths <- list()
    for (f in CHATTERBOX_TURBO_FILES) {
        name <- tools::file_path_sans_ext(basename(f))
        tryCatch({
            paths[[name]] <- do.call(hfhub::hub_download,
                                     c(list(CHATTERBOX_TURBO_REPO, f,
                                            local_files_only = TRUE), rev))
        }, error = function(e) {
            stop(
                 "Turbo model file '", f, "' not found. ",
                 "Run download_chatterbox_turbo_models() first.",
                 call. = FALSE
            )
        })
    }

    paths
}
