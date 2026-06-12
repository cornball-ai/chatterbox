# torch GC tuning for inference loops

#' Recommended torch garbage-collection settings for chatterbox
#'
#' torch's allocators invoke a full R garbage collection based on
#' settings that are read ONCE at torch startup. The default CUDA
#' trigger (collections begin once torch reserves 20 percent of the
#' card) sits below chatterbox's ~4.6 GB loaded footprint on most GPUs,
#' which makes autoregressive inference collection-bound: ~91 percent
#' of pure-R generation wall time is GC, and even compiled-loop backends
#' are throttled by it (their allocations flow through the same
#' allocator). With the trigger line above the model floor, pure R runs
#' ~10x faster and the jit backend ~15x.
#'
#' Only one option matters for speed:
#' \code{torch.cuda_allocator_reserved_rate}. Sweeps over 0.3-0.8 all
#' give identical speed; the value just chooses how high the VRAM
#' plateau sits. \code{torch.cuda_allocator_allocated_rate} (default
#' 0.8) caps that plateau and can be lowered to ~0.6 on shared GPUs at
#' no speed cost. The remaining settings
#' (\code{torch.threshold_call_gc},
#' \code{torch.cuda_allocator_allocated_reserved_rate}) measured as not
#' worth touching.
#'
#' This helper does not (and cannot) change the settings for the current
#' session: torch reads them at initialization, so they belong in your
#' .Rprofile or at the very top of a script, before torch loads. Printing
#' the returned object shows the exact snippet for this machine; the
#' helper warns when torch is already initialized. Scripts can apply the
#' values directly with \code{do.call(options, chatterbox_gc_options())}
#' (again: before torch loads).
#'
#' Rule of thumb for loops: collect once per utterance, not thousands of
#' times inside it. \code{\link{tts_chunked}} does this automatically;
#' in your own batch loops, call \code{gc()} after each
#' \code{generate()}.
#'
#' @param vram_gb Total GPU memory in GB. Default: detected via
#'   nvidia-smi, falling back to 16.
#' @return A named list of the recommended \code{options()} values,
#'   classed \code{"chatterbox_gc_options"} so it prints as the full
#'   tuning advice for this machine.
#' @examples
#' chatterbox_gc_options(vram_gb = 16)
#' @export
chatterbox_gc_options <- function(vram_gb = NULL) {
    if (is.null(vram_gb)) {
        smi <- suppressWarnings(tryCatch(
                system2("nvidia-smi",
                        c("--query-gpu=memory.total",
                          "--format=csv,noheader,nounits"),
                        stdout = TRUE, stderr = FALSE),
                error = function(e) character(0)
            ))
        vram_gb <- if (length(smi) >= 1 && nzchar(smi[1]) &&
            !is.na(suppressWarnings(as.numeric(smi[1])))) {
            round(as.numeric(smi[1]) / 1024, 1)
        } else {
            16
        }
    }

    # The trigger line (a fraction of total VRAM) must clear what the
    # loaded model reserves (~3.6-4.6 GB depending on card); the exact
    # value past that only moves the VRAM plateau, not speed. Tiers:
    # 16 GB and 6 GB measured, 8 GB projected from the rule.
    rate <- if (vram_gb <= 6.5) 0.75 else if (vram_gb <= 10) 0.6 else 0.5

    opts <- list(torch.cuda_allocator_reserved_rate = rate)

    if (isNamespaceLoaded("torch")) {
        warning("torch is already initialized in this session; these ",
                "options take effect only in a fresh R session that sets ",
                "them before torch loads.", call. = FALSE)
    }

    structure(opts, vram_gb = vram_gb, class = "chatterbox_gc_options")
}

#' Print method for chatterbox_gc_options
#'
#' @param x Object from \code{\link{chatterbox_gc_options}}
#' @param ... Ignored
#' @return \code{x}, invisibly
#' @export
print.chatterbox_gc_options <- function(x, ...) {
    vram_gb <- attr(x, "vram_gb")
    rate <- x$torch.cuda_allocator_reserved_rate

    cat("Recommended for a ", vram_gb,
        " GB GPU - put this in .Rprofile or\n",
        "at the top of your script, BEFORE torch loads:\n\n", sep = "")
    cat(sprintf("    options(torch.cuda_allocator_reserved_rate = %.2f)\n\n",
                rate))
    if (vram_gb >= 8) {
        cat("Optional, to hold the VRAM plateau lower (e.g. shared GPUs),\n",
            "at no speed cost:\n\n", sep = "")
        cat("    options(torch.cuda_allocator_allocated_rate = 0.6)\n\n")
    }
    cat("In batch loops, call gc() after each generate().\n")

    if (vram_gb <= 6.5) {
        cat("\nNote: on a ", vram_gb, " GB card the model floor leaves",
            " little headroom, so the\n0.8 backstop still fires some",
            " collections: expect ~3-5x from tuning for\npure R, not",
            " the ~10x larger cards see. traced = TRUE measured",
            " fastest\non 6 GB hardware (88-94 ms/token, ~5 GB peak -",
            " tight but it fits).\nDo NOT lower allocated_rate here -",
            " 60% of a small card sits below\nthe model floor and",
            " recreates the constant-collection regime.\n", sep = "")
    }

    invisible(x)
}

