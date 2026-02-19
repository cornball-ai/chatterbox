#' Model Download Utilities
#'
#' Download Chatterbox models from HuggingFace using hfhub.
#' Requires explicit download with user consent (no auto-download).

CHATTERBOX_REPO <- "ResembleAI/chatterbox"

CHATTERBOX_FILES <- c(
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt"
)

# Approximate total model size in MB
.model_size_mb <- 2000

CHATTERBOX_TURBO_REPO <- "ResembleAI/chatterbox-turbo"

CHATTERBOX_TURBO_FILES <- c(
    "t3_turbo_v1.safetensors",
    "s3gen_meanflow.safetensors",
    "s3gen.safetensors",
    "ve.safetensors",
    "conds.pt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json"
)

# Approximate turbo model size in MB
.turbo_model_size_mb <- 3800

#' Check if Models are Downloaded
#'
#' @return TRUE if all model files exist locally
#' @export
#' @examples
#' models_available()
models_available <- function ()
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        return(FALSE)
    }

    tryCatch({
        for (f in CHATTERBOX_FILES) {
            hfhub::hub_download(CHATTERBOX_REPO, f, local_files_only = TRUE)
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
#' @return Named list of local file paths (invisibly)
#' @export
#' @examples
#' \dontrun{
#' # Download models (~2GB)
#' download_chatterbox_models()
#' }
download_chatterbox_models <- function (force = FALSE)
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install with: install.packages('hfhub')")
    }

    # Check if already downloaded
    if (!force && models_available()) {
        message("Chatterbox models are already downloaded.")
        return(invisible(get_model_paths()))
    }

    # Ask for consent (required for CRAN compliance)
    # Skip prompt if chatterbox.consent option is set
    if (isTRUE(getOption("chatterbox.consent"))) {
        # Consent already given programmatically
    } else if (interactive()) {
        ans <- utils::askYesNo(
            paste0("Download Chatterbox models (~", .model_size_mb, " MB) from HuggingFace?"),
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
            path <- hfhub::hub_download(CHATTERBOX_REPO, f, force_download = force)
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
#' @return Named list of local file paths
#' @export
get_model_paths <- function ()
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install with: install.packages('hfhub')")
    }

    paths <- list()
    for (f in CHATTERBOX_FILES) {
        name <- tools::file_path_sans_ext(basename(f))
        tryCatch({
            paths[[name]] <- hfhub::hub_download(CHATTERBOX_REPO, f, local_files_only = TRUE)
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
#' @return TRUE if all turbo model files exist locally
#' @export
#' @examples
#' turbo_models_available()
turbo_models_available <- function ()
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        return(FALSE)
    }

    tryCatch({
        for (f in CHATTERBOX_TURBO_FILES) {
            hfhub::hub_download(CHATTERBOX_TURBO_REPO, f, local_files_only = TRUE)
        }
        TRUE
    }, error = function(e) FALSE)
}

#' Download Chatterbox Turbo Models from HuggingFace
#'
#' Download all Chatterbox Turbo model files from HuggingFace.
#' In interactive sessions, asks for user consent before downloading.
#' The turbo model uses flow matching for faster inference.
#'
#' @param force Re-download even if files exist
#' @return Named list of local file paths (invisibly)
#' @export
#' @examples
#' \dontrun{
#' # Download turbo models (~3.8GB)
#' download_chatterbox_turbo_models()
#' }
download_chatterbox_turbo_models <- function (force = FALSE)
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install with: install.packages('hfhub')")
    }

    if (!force && turbo_models_available()) {
        message("Chatterbox Turbo models are already downloaded.")
        return(invisible(get_turbo_model_paths()))
    }

    if (isTRUE(getOption("chatterbox.consent"))) {
        # Consent already given programmatically
    } else if (interactive()) {
        ans <- utils::askYesNo(
            paste0("Download Chatterbox Turbo models (~", .turbo_model_size_mb,
                   " MB) from HuggingFace?"),
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
            path <- hfhub::hub_download(CHATTERBOX_TURBO_REPO, f,
                                         force_download = force)
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
#' @return Named list of local file paths
#' @export
get_turbo_model_paths <- function ()
{
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("hfhub package required. Install with: install.packages('hfhub')")
    }

    paths <- list()
    for (f in CHATTERBOX_TURBO_FILES) {
        name <- tools::file_path_sans_ext(basename(f))
        tryCatch({
            paths[[name]] <- hfhub::hub_download(CHATTERBOX_TURBO_REPO, f,
                                                   local_files_only = TRUE)
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
