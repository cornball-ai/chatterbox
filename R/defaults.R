# Hardware-adaptive defaults: GC settings, backend, and chunking
# thresholds per detected GPU/CPU. Measured tiers: 16 GB (RTX 5060 Ti)
# and 6 GB (GTX 1660 Ti); 8/12 GB projected from the tier rule.

#' Recommended chatterbox settings for this machine
#'
#' Detects the GPU (or its absence) and returns everything worth setting
#' for it: the torch GC options (which must be set BEFORE torch loads -
#' see \code{\link{chatterbox_gc_options}} for why), the fastest
#' validated backend, the per-call token budget, and when to switch to
#' \code{\link{tts_chunked}}. Printing the result shows a ready-to-paste
#' setup snippet.
#'
#' Measured tiers (long-form, tuned GC): 16 GB RTX 5060 Ti - jit
#' 11 ms/token, container parity; 6 GB GTX 1660 Ti - jit 35-38 ms/token
#' vs container 30, in 4.7 GB VRAM. The 8 and 12 GB tiers are projected
#' from the rule (the GC trigger line must clear the ~4.6 GB loaded
#' model) and marked as such when printed.
#'
#' @param vram_gb Total GPU memory in GB. Default: detected via
#'   nvidia-smi; 0 (or detection failure) means CPU-only.
#' @return An object of class \code{"chatterbox_defaults"}: a list with
#'   \code{device}, \code{vram_gb}, \code{options} (for
#'   \code{do.call(options, ...)} before torch loads), \code{backend},
#'   \code{max_new_tokens}, \code{chunk_chars}, and \code{measured}.
#' @examples
#' chatterbox_defaults(vram_gb = 6)
#' chatterbox_defaults(vram_gb = 0) # CPU
#' @export
chatterbox_defaults <- function (vram_gb = NULL) {
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
            0
        }
    }

    if (vram_gb < 4) {
        # CPU (or a card too small for the ~4.6 GB loaded model).
        # The CUDA allocator knobs are irrelevant; only the CPU
        # allocation odometer exists, and it measured as minor.
        out <- list(
            device = "cpu",
            vram_gb = vram_gb,
            options = list(),
            backend = "r",
            max_new_tokens = 1000L,
            chunk_chars = 200L,
            measured = FALSE
        )
    } else {
        rate <- if (vram_gb <= 6.5) 0.75 else if (vram_gb <= 10) 0.6 else 0.5
        out <- list(
            device = "cuda",
            vram_gb = vram_gb,
            options = list(torch.cuda_allocator_reserved_rate = rate),
            backend = "jit",
            max_new_tokens = 1000L,
            chunk_chars = 200L,
            measured = vram_gb <= 6.5 || vram_gb > 12
        )
    }

    if (isNamespaceLoaded("torch") && length(out$options) > 0) {
        warning("torch is already initialized in this session; the GC ",
            "options take effect only in a fresh R session that sets ",
            "them before torch loads.", call. = FALSE)
    }

    structure(out, class = "chatterbox_defaults")
}

#' Print method for chatterbox_defaults
#'
#' @param x Object from \code{\link{chatterbox_defaults}}
#' @param ... Ignored
#' @return \code{x}, invisibly
#' @export
print.chatterbox_defaults <- function (x, ...) {
    if (x$device == "cpu") {
        cat("CPU-only setup (no usable GPU detected).\n\n",
            "    library(chatterbox)\n",
            "    model <- load_chatterbox(chatterbox(\"cpu\"))\n\n",
            "Use backend = \"r\". Expect minutes per utterance; for\n",
            "anything longer than a sentence or two, use tts_chunked()\n",
            "so audio arrives incrementally.\n", sep = "")
        return(invisible(x))
    }

    tier <- if (isTRUE(x$measured)) "measured" else "projected"
    rate <- x$options$torch.cuda_allocator_reserved_rate
    cat(sprintf("Recommended for a %s GB GPU (%s tier) - put the\n",
        format(x$vram_gb), tier))
    cat("options() line in .Rprofile or at the top of your script,\n")
    cat("BEFORE torch loads:\n\n")
    cat(sprintf("    options(torch.cuda_allocator_reserved_rate = %.2f)\n",
        rate))
    cat("    library(chatterbox)\n")
    cat("    model <- load_chatterbox(chatterbox(\"cuda\"))\n")
    cat(sprintf(
        "    result <- generate(model, text, voice, backend = \"%s\")\n\n",
        x$backend))
    cat(sprintf(
        "Per call, up to max_new_tokens = %d (~40 s of audio). For\n",
        x$max_new_tokens))
    cat(sprintf(
        "longer texts use tts_chunked() (sentence chunks, ~%d chars,\n",
        x$chunk_chars))
    cat("one gc() per chunk). In your own batch loops, call gc() after\n")
    cat("each generate().\n")

    if (x$vram_gb <= 6.5) {
        cat("\nNote: on a ", format(x$vram_gb), " GB card the model floor",
            " leaves little headroom,\nso the 0.8 backstop still fires",
            " some collections. Measured on a\nGTX 1660 Ti: jit",
            " 35-38 ms/token (~4.7 GB peak) vs container 30;\npure R",
            " ~10x slower. Do NOT lower",
            " torch.cuda_allocator_allocated_rate\nhere - 60% of a small",
            " card sits below the model floor and recreates\nthe",
            " constant-collection regime.\n", sep = "")
    } else if (x$vram_gb >= 8) {
        cat("\nOptional, to hold the VRAM plateau lower (e.g. shared",
            " GPUs), at\nno speed cost:\n\n",
            "    options(torch.cuda_allocator_allocated_rate = 0.6)\n",
            sep = "")
    }

    invisible(x)
}
