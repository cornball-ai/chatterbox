# torch GC tuning for inference loops

#' Recommended torch garbage-collection settings for chatterbox
#'
#' torch's allocators invoke a full R garbage collection based on four
#' settings that are read ONCE at torch startup. The defaults are tuned
#' for small GPUs and make autoregressive inference collection-bound:
#' profiling shows ~91 percent of pure-R generation wall time spent in
#' GC, and tuned settings run the same generation ~12x faster.
#'
#' This helper does not (and cannot) change the settings for the current
#' session: torch reads them at initialization, so they belong in your
#' .Rprofile or at the very top of a script, before torch loads. It
#' prints the exact snippet for this machine and warns when torch is
#' already initialized.
#'
#' The two untouched companion settings
#' (\code{torch.cuda_allocator_allocated_rate} and
#' \code{torch.cuda_allocator_allocated_reserved_rate}, both 0.8) are
#' the self-regulation backstop that forces collections near the top of
#' the card; leave them at their defaults.
#'
#' Rule of thumb for loops: collect once per utterance, not thousands of
#' times inside it. \code{\link{tts_chunked}} does this automatically;
#' in your own batch loops, call \code{gc()} after each
#' \code{generate()}.
#'
#' @param vram_gb Total GPU memory in GB. Default: detected via
#'   nvidia-smi, falling back to 16.
#' @return Invisibly, a named list of the recommended options.
#' @examples
#' chatterbox_gc_options(vram_gb = 16)
#' @export
chatterbox_gc_options <- function (vram_gb = NULL) {
    if (is.null(vram_gb)) {
        smi <- suppressWarnings(tryCatch(
            system2("nvidia-smi",
                c("--query-gpu=memory.total", "--format=csv,noheader,nounits"),
                stdout = TRUE, stderr = FALSE),
            error = function (e) character(0)
        ))
        vram_gb <- if (length(smi) >= 1 && nzchar(smi[1]) &&
            !is.na(suppressWarnings(as.numeric(smi[1])))) {
            round(as.numeric(smi[1]) / 1024, 1)
        } else {
            16
        }
    }

    # The collection trigger line (a fraction of total VRAM) must clear
    # the ~3.2GB fp32 model floor with headroom, but stay under the 0.8
    # allocated-rate backstop. Measured: 0.5 on 16GB, 0.75 on 6GB.
    rate <- round(min(max(4.5 / vram_gb, 0.5), 0.8), 2)

    opts <- list(
        torch.cuda_allocator_reserved_rate = rate,
        torch.threshold_call_gc = 16000
    )

    cat("Recommended for a ", vram_gb, " GB GPU - put this in .Rprofile or\n",
        "at the top of your script, BEFORE torch loads:\n\n", sep = "")
    cat(sprintf(paste0(
        "    options(\n",
        "        torch.cuda_allocator_reserved_rate = %.2f,\n",
        "        torch.threshold_call_gc = 16000\n",
        "    )\n\n"), rate))
    cat("In batch loops, call gc() after each generate().\n")

    if (vram_gb <= 6.5) {
        cat("\nNote: on a ", vram_gb, " GB card the fp32 model floor",
            " (~3.2 GB) leaves little\nheadroom; the pure-R backend",
            " is usually the practical choice there\n(traced graphs",
            " add ~1 GB of weight copies).\n", sep = "")
    }

    if (isNamespaceLoaded("torch")) {
        warning("torch is already initialized in this session; these ",
            "options take effect only in a fresh R session that sets ",
            "them before torch loads.", call. = FALSE)
    }

    invisible(opts)
}
