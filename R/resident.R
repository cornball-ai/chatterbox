# In-process model residency: pinned host weights, disposable GPU copies.
#
# A resident model keeps its canonical weights as page-locked (pinned) CPU
# tensors for the life of the handle. Activation creates the GPU
# representation with a fast DMA copy from pinned memory; deactivation
# destroys only the GPU representation and repoints the modules at the
# pinned host storage. Reactivation never touches the disk, so swapping
# chatterbox in and out of a small GPU is a sub-second operation instead
# of a full reload.
#
# Same design as whisper's R/resident.R (the two packages present one
# contract to a residency broker), adapted to chatterbox's shape: the
# model is a classed list holding THREE nn_modules (voice_encoder, t3,
# s3gen -- s3gen includes the vocoder), so the manifest and the pinned
# set use flat module-prefixed names ("t3.<name>", ...). The tokenizer is
# a plain R object with no tensors and rides along untouched.
#
# Mechanics rest on two torch behaviours (verified against the installed
# version in whisper's port, re-verified by this package's tests):
# - nn_module$to() REBINDS parameter/buffer objects, so pinned host
#   tensors held in res$pinned survive activation -- and any tensor
#   handle taken before a transition is stale after it. All rebinding
#   therefore resolves the modules' CURRENT tensors by name, every time.
# - Tensor$set_data() works across devices: a CUDA parameter can be
#   repointed directly at a pinned CPU tensor. That is the evict
#   mechanism; the orphaned CUDA storage is reclaimed by gc() +
#   cuda_empty_cache().
#
# States: inactive -> activating -> active -> deactivating -> inactive.
# Failed transitions roll back to pinned host state; a rollback that
# cannot be verified leaves the handle "broken" (fail-closed: only status
# and unload work). "unloaded" is terminal.

#' Format a byte count for messages
#' @noRd
.fmt_bytes <- function(b) {
    if (is.null(b) || is.na(b) || b <= 0) {
        return("0 B")
    }
    u <- c("B", "KB", "MB", "GB", "TB")
    i <- min(floor(log(b, 1024)), 4)
    sprintf("%.2f %s", b / 1024^i, u[i + 1])
}

#' The model's tensor-bearing modules, by name
#' @noRd
.resident_modules <- function(model) {
    mods <- list(voice_encoder = model$voice_encoder, t3 = model$t3,
                 s3gen = model$s3gen)
    mods[!vapply(mods, is.null, logical(1))]
}

#' Enumerate current named tensors across all modules, module-prefixed
#'
#' Always re-enumerated at the point of use: module$to() rebinds tensor
#' objects, so handles from load time go stale after every transition.
#' @noRd
.resident_tensors <- function(model) {
    out <- list()
    mods <- .resident_modules(model)
    for (mn in names(mods)) {
        ts <- c(mods[[mn]]$named_parameters(), mods[[mn]]$named_buffers())
        names(ts) <- paste0(mn, ".", names(ts))
        out <- c(out, ts)
    }
    out
}

#' Manifest of the model's tensors: name, kind, dtype, shape, logical bytes
#' @noRd
.resident_manifest <- function(model) {
    ts <- .resident_tensors(model)
    pn <- unlist(lapply(names(.resident_modules(model)), function(mn) {
        paste0(mn, ".",
               names(.resident_modules(model)[[mn]]$named_parameters()))
    }))
    data.frame(
        name = names(ts),
        kind = ifelse(names(ts) %in% pn, "parameter", "buffer"),
        dtype = vapply(ts, function(t) as.character(t$dtype), character(1)),
        shape = vapply(ts, function(t) paste(dim(t), collapse = "x"),
                       character(1)),
        bytes = vapply(ts, function(t) t$numel() * t$element_size(),
                       numeric(1)),
        stringsAsFactors = FALSE, row.names = NULL
    )
}

#' Require exact manifest equality before a transition
#' @noRd
.resident_check <- function(model, manifest) {
    cur <- .resident_tensors(model)
    if (!setequal(names(cur), manifest$name)) {
        stop("model tensors no longer match the residency manifest: ",
             "expected ", nrow(manifest), " names, found ", length(cur))
    }
    for (i in seq_len(nrow(manifest))) {
        t <- cur[[manifest$name[i]]]
        if (as.character(t$dtype) != manifest$dtype[i]) {
            stop("dtype changed for '", manifest$name[i], "': ",
                 as.character(t$dtype), " vs manifest ", manifest$dtype[i])
        }
        if (paste(dim(t), collapse = "x") != manifest$shape[i]) {
            stop("shape changed for '", manifest$name[i], "'")
        }
    }
    invisible(cur)
}

#' TRUE when every manifest tensor sits on exactly the target device
#' @noRd
.resident_on_target <- function(res) {
    ok <- tryCatch({
        cur <- .resident_tensors(res$model)
        if (!setequal(names(cur), res$manifest$name)) {
            FALSE
        } else {
            all(vapply(cur, function(t) t$device == res$target_device,
                       logical(1)))
        }
    }, error = function(e) FALSE)
    isTRUE(ok)
}

#' TRUE when every manifest tensor is back on CPU (pinned host state)
#'
#' Pinnedness itself is a load-time invariant: the tensors in res$pinned
#' were verified pinned once at load, a pinned allocation cannot silently
#' become pageable, and rollback rebinds to exactly those tensors by name.
#' (Per-tensor is_pinned() is not used here: this torch build requires the
#' deprecated device argument, and libtorch emits a stderr warning per
#' call.)
#' @noRd
.resident_verify_pinned <- function(res) {
    ok <- tryCatch({
        cur <- .resident_tensors(res$model)
        if (!setequal(names(cur), res$manifest$name)) {
            FALSE
        } else {
            all(vapply(cur, function(t) t$device$type == "cpu", logical(1)))
        }
    }, error = function(e) FALSE)
    isTRUE(ok)
}

#' Synchronize the handle's own GPU, never whichever device is current
#' @noRd
.resident_sync <- function(res) {
    torch::cuda_synchronize(device = res$target_device$index)
}

#' Rebind every current tensor to its pinned host copy and reclaim VRAM
#'
#' Shared workhorse of deactivation and of rollback after a failed
#' activation. Returns TRUE only when the restored pinned state has been
#' verified tensor-by-tensor. Synchronization and cache-release failures
#' make the rollback unprovable -- an asynchronous CUDA failure must not
#' let a handle return to "inactive" -- and the cause is appended to
#' last_error so it survives into the broken state.
#' @noRd
.resident_rollback <- function(res, release = TRUE) {
    cause <- NULL
    ok <- tryCatch({
        .resident_sync(res)
        cur <- .resident_tensors(res$model)
        if (!setequal(names(cur), names(res$pinned))) {
            cause <- "pinned set does not cover the current tensors"
            FALSE
        } else {
            torch::with_no_grad({
                for (nm in names(cur)) {
                    cur[[nm]]$set_data(res$pinned[[nm]])
                }
            })
            gc()
            if (isTRUE(release)) {
                torch::cuda_empty_cache()
            }
            TRUE
        }
    }, error = function(e) {
        cause <<- conditionMessage(e)
        FALSE
    })
    if (!isTRUE(ok)) {
        res$last_error <- paste(
            c(res$last_error, paste0("rollback: ", cause %||% "unknown")),
            collapse = "; ")
        return(FALSE)
    }
    verified <- .resident_verify_pinned(res)
    if (verified) {
        # The weights are host-side again, so model$device -- which
        # generate() uses to place inputs -- must say so. Only on a
        # verified restore: a handle heading for "broken" keeps the
        # device string it had, since where its tensors live is exactly
        # what could not be established.
        res$model$device <- "cpu"
    }
    verified
}

#' Manifest bytes currently observed on the GPU, or NA when unknowable
#' @noRd
.resident_gpu_bytes_observed <- function(res) {
    tryCatch({
        cur <- .resident_tensors(res$model)
        if (!setequal(names(cur), res$manifest$name)) {
            NA_real_
        } else {
            sum(vapply(cur, function(t) {
                if (t$device$type == "cuda") t$numel() * t$element_size()
                else 0
            }, numeric(1)))
        }
    }, error = function(e) NA_real_)
}

#' Default move seam: one non-blocking move per module
#'
#' Kept as a seam (stored on the handle) so tests can inject a partial
#' move that fails midway and prove the rollback path.
#' @noRd
.resident_move <- function(res, device) {
    for (m in .resident_modules(res$model)) {
        m$to(device = device, non_blocking = TRUE)
    }
    invisible(NULL)
}

#' Refuse verbs on handles that are terminal or fail-closed
#' @noRd
.resident_guard <- function(res, verb) {
    if (!inherits(res, "chatterbox_resident")) {
        stop("not a chatterbox_resident handle")
    }
    if (identical(res$state, "unloaded")) {
        stop(verb, "(): handle has been unloaded")
    }
    if (identical(res$state, "broken")) {
        stop(verb, "(): handle is broken (",
             res$last_error %||% "unknown error",
             "); only resident_status() and resident_unload() are available")
    }
    invisible(res)
}

#' HF snapshot revision from an hfhub cache path, or NA
#'
#' Parses the path as given. normalizePath() must NOT be used here: the
#' hfhub layout stores snapshots/<revision>/<file> as a symlink into
#' blobs/<hash>, so resolving symlinks erases the very path segment this
#' function exists to read.
#' @noRd
.snapshot_revision <- function(path) {
    parts <- strsplit(path.expand(path), "/", fixed = TRUE)[[1]]
    i <- which(parts == "snapshots")
    if (length(i) == 1 && length(parts) > i) parts[i + 1L] else NA_character_
}

#' Load Chatterbox as a Resident (Pinned Host Weights)
#'
#' Loads the model (standard or turbo) and retains its canonical weights
#' as page-locked (pinned) CPU tensors, so the GPU representation can be
#' created and destroyed repeatedly without reloading from disk.
#' [resident_activate()] copies the weights to the GPU (a DMA
#' transfer from pinned memory); [resident_deactivate()] frees the
#' GPU copy and repoints the model at the pinned host storage. The handle
#' starts inactive.
#'
#' Weights stay at their native dtypes (the validated numerics are fp32);
#' nothing is converted. The torch allocator GC options are applied before
#' the first CUDA operation, exactly as [chatterbox()] does --
#' without them, both inference and the activation copy itself become
#' collection-bound on cards where the model exceeds 20 percent of VRAM.
#'
#' @param turbo Load the turbo variant (default FALSE).
#' @param device Target CUDA device for activation (default "cuda").
#'   Residency requires CUDA; pinned host memory exists to feed it. A bare
#'   "cuda" is resolved to the current device's explicit index at load
#'   time, and the handle stays bound to that exact device (`"cuda:N"`)
#'   for every later transition and synchronize.
#' @param tune_gc Apply the allocator GC options before CUDA init
#'   (default TRUE), as [chatterbox()] does.
#' @param verbose Print progress messages.
#'
#' @return A `chatterbox_resident` handle (an environment). Fields of
#'   interest via [resident_status()]: state, byte counts, and a
#'   content identity (per-artifact sha256, HF repo and snapshot revision,
#'   variant).
#'
#' @examples
#' \dontrun{
#' res <- resident_load(turbo = TRUE)
#' resident_activate(res)
#' out <- resident_generate(res, "Hello from a resident model.",
#'     system.file("audio", "jfk.wav", package = "chatterbox"))
#' resident_deactivate(res) # VRAM freed, weights stay pinned in RAM
#' resident_activate(res) # fast: DMA copy, no disk
#' resident_unload(res)
#' }
#' @export
resident_load <- function(turbo = FALSE, device = "cuda", tune_gc = TRUE,
                          verbose = TRUE) {
    # GC tuning must run before ANY CUDA op: torch reads the allocator
    # rates once, at lazy CUDA init, and the cuda_is_available() probe
    # below is enough to trigger it. This mirrors chatterbox()'s ordering
    # (tune, then probe) and is not cosmetic -- on a card where the model
    # exceeds the default 20% reserved-rate floor, missing it makes both
    # inference and the activation copy collection-bound. It takes the raw
    # device string: .set_cuda_gc_options() parses the index itself and
    # reads VRAM via nvidia-smi, so it needs no CUDA context.
    # Accept a string or a torch_device, as whisper's resident_load() does.
    # as.character() on a torch_device is pure formatting -- no CUDA context
    # -- so the raw string is available before the probe below.
    dev_str <- if (inherits(device, "torch_device")) {
        as.character(device)
    } else {
        device
    }
    if (!is.character(dev_str) || length(dev_str) != 1L) {
        stop("device must be a device string or torch_device, e.g. \"cuda\"")
    }
    if (!grepl("^cuda", dev_str)) {
        stop("resident_load() requires a CUDA target device")
    }
    if (isTRUE(tune_gc)) {
        .set_cuda_gc_options(dev_str, turbo)
    }

    if (!torch::cuda_is_available()) {
        stop("CUDA is not available")
    }
    target <- torch::torch_device(dev_str)
    if (is.null(target$index)) {
        target <- torch::torch_device(
            paste0("cuda:", torch::cuda_current_device()))
    }

    # Build on CPU; the device move is what resident_activate() is for.
    # chatterbox()/load_chatterbox() narrate their own progress, so
    # verbose = FALSE has to suppress theirs too, not just ours.
    load_model <- function() {
        chatterbox(device = "cpu", turbo = turbo, load = TRUE,
                   tune_gc = FALSE)
    }
    model <- if (isTRUE(verbose)) load_model() else
        suppressMessages(load_model())

    manifest <- .resident_manifest(model)

    if (verbose) {
        message("Pinning ", nrow(manifest), " tensors (",
                .fmt_bytes(sum(manifest$bytes)), ") in host memory")
    }
    pinned <- list()
    torch::with_no_grad({
        ts <- .resident_tensors(model)
        for (nm in names(ts)) {
            t <- ts[[nm]]
            pinned[[nm]] <- t$detach()$pin_memory(target)
            t$set_data(pinned[[nm]])
        }
    })
    # The pageable originals lost their last reference in set_data; drop
    # them so the pinned copies are the model's only host representation.
    gc()

    # Verify pinnedness once, here, where it is established. This is the
    # only place is_pinned() runs outside the tests: the R binding
    # requires the deprecated device argument, so each call draws a
    # libtorch stderr warning -- a bounded burst at load.
    for (nm in names(pinned)) {
        if (!pinned[[nm]]$is_pinned(target)) {
            stop("pin_memory() did not produce a pinned tensor for '",
                 nm, "'")
        }
    }

    paths <- if (isTRUE(turbo)) get_turbo_model_paths() else
        get_model_paths()
    if (verbose) message("Hashing ", length(paths), " artifacts (sha256)")
    artifacts <- lapply(names(paths), function(nm) {
        list(name = nm, file = basename(paths[[nm]]),
             bytes = unname(file.size(paths[[nm]])),
             sha256 = unname(tools::sha256sum(paths[[nm]])))
    })
    names(artifacts) <- names(paths)
    # Reported, not assumed: chatterbox loads at the weights' native dtype
    # (fp32 in practice), so read it off the manifest rather than hardcode.
    par_dtypes <- manifest$dtype[manifest$kind == "parameter"]
    identity <- list(
        variant = if (isTRUE(turbo)) "turbo" else "standard",
        repo = if (isTRUE(turbo)) CHATTERBOX_TURBO_REPO else CHATTERBOX_REPO,
        revision = .snapshot_revision(paths[[1]]),
        dtype = if (length(par_dtypes)) {
            names(sort(table(par_dtypes), decreasing = TRUE))[1]
        } else {
            NA_character_
        },
        artifacts = artifacts
    )

    res <- new.env(parent = emptyenv())
    res$model <- model
    res$pinned <- pinned
    res$manifest <- manifest
    res$state <- "inactive"
    res$in_flight <- FALSE
    res$identity <- identity
    res$paths <- paths
    res$target_device <- target
    res$pinned_bytes <- sum(manifest$bytes)
    res$gpu_bytes <- 0
    res$move <- .resident_move
    res$last_error <- NULL
    class(res) <- "chatterbox_resident"
    res
}

#' Activate a Resident Model (Create the GPU Representation)
#'
#' Copies the pinned host weights to the target device in one synchronous
#' pass (non-blocking per-tensor copies, then a device synchronize). The
#' transition is transactional: on any failure -- including a partial
#' move after an out-of-memory error -- every tensor is rebound to its
#' pinned host copy, the CUDA allocation is released, and the handle
#' returns to "inactive". A rollback that cannot be verified leaves the
#' handle "broken", where only [resident_status()] and
#' [resident_unload()] operate.
#'
#' @param res A `chatterbox_resident` handle from [resident_load()].
#' @return The handle, invisibly. No-op when already active.
#' @export
resident_activate <- function(res) {
    .resident_guard(res, "resident_activate")
    if (identical(res$state, "active")) {
        return(invisible(res))
    }
    if (!identical(res$state, "inactive")) {
        stop("cannot activate from state '", res$state, "'")
    }
    # Pre-move manifest check: abort while still inactive, nothing moved.
    .resident_check(res$model, res$manifest)

    res$state <- "activating"
    err <- NULL
    ok <- tryCatch({
        res$move(res, res$target_device)
        .resident_sync(res)
        TRUE
    }, error = function(e) {
        err <<- conditionMessage(e)
        FALSE
    })

    if (ok && .resident_on_target(res)) {
        # generate()/create_voice_embedding() place their input tensors by
        # model$device (tts.R), so it must track where the weights
        # actually are -- set only after the move is verified, and reset
        # by every path that returns the weights to host memory.
        res$model$device <- as.character(res$target_device)
        res$state <- "active"
        res$gpu_bytes <- res$pinned_bytes
        return(invisible(res))
    }
    if (is.null(err)) err <- "device verification failed after move"
    res$last_error <- err

    if (.resident_rollback(res)) {
        res$state <- "inactive"
        res$gpu_bytes <- 0
        stop("resident_activate() failed (rolled back to pinned host ",
             "state): ", err)
    }
    res$state <- "broken"
    res$gpu_bytes <- .resident_gpu_bytes_observed(res)
    stop("resident_activate() failed and rollback could not be verified; ",
         "handle is broken: ", res$last_error)
}

#' Deactivate a Resident Model (Destroy the GPU Representation)
#'
#' Rebinds every tensor to its pinned host copy, releases the orphaned
#' CUDA storage, and verifies the restored state tensor-by-tensor. The
#' pinned weights remain in host memory, so a later
#' [resident_activate()] is a DMA copy, not a disk reload.
#' Refuses while a generation is in flight.
#'
#' `release` decides who gets the freed VRAM, and it is worth about an
#' order of magnitude. With `release = TRUE` (the default) the CUDA
#' caching allocator hands its blocks back to the driver, so the memory
#' is visible as free to other processes -- but the next activation must
#' re-acquire every block from the driver, which on a small card measured
#' ~9 ms per tensor (2.85 GB across 948 tensors: 9.2 s, 0.31 GB/s).
#' With `release = FALSE` the blocks stay in this process's pool for the
#' next model to reuse, and the same activation took 0.86 s (3.29 GB/s),
#' matching raw pinned-DMA bandwidth. Use `FALSE` when one process hosts
#' every model and switches between them; use the default when a
#' different process needs the card.
#'
#' @param res A `chatterbox_resident` handle.
#' @param release Return the allocator's blocks to the driver (default
#'   TRUE). FALSE keeps them pooled for a fast next activation; the
#'   weights are freed either way, and `gpu_bytes` goes to zero in
#'   both cases (the retained pool is process overhead, attributable to
#'   no model).
#' @return The handle, invisibly. No-op when already inactive.
#' @export
resident_deactivate <- function(res, release = TRUE) {
    .resident_guard(res, "resident_deactivate")
    if (isTRUE(res$in_flight)) {
        stop("resident_deactivate(): a generation is in flight")
    }
    if (identical(res$state, "inactive")) {
        return(invisible(res))
    }
    if (!identical(res$state, "active")) {
        stop("cannot deactivate from state '", res$state, "'")
    }
    res$state <- "deactivating"
    if (.resident_rollback(res, release = release)) {
        res$state <- "inactive"
        res$gpu_bytes <- 0
        return(invisible(res))
    }
    res$state <- "broken"
    res$gpu_bytes <- .resident_gpu_bytes_observed(res)
    stop("resident_deactivate() could not verify the pinned host state; ",
         "handle is broken",
         if (is.null(res$last_error)) "" else paste0(": ", res$last_error))
}

#' Generate Speech with a Resident Model
#'
#' Runs [generate()] through the resident handle's model.
#' Requires the model to be active; marks the handle in-flight for the
#' duration so a concurrent deactivation is refused.
#'
#' @param res A `chatterbox_resident` handle, currently active.
#' @param text Text to synthesize.
#' @param voice Reference voice (audio path or voice file), per
#'   [generate()].
#' @param ... Passed to [generate()]: `exaggeration`,
#'   `cfg_weight`, `temperature`, `backend`, ...
#' @return The generation result, same shape as [generate()].
#' @export
resident_generate <- function(res, text, voice, ...) {
    .resident_guard(res, "resident_generate")
    if (!identical(res$state, "active")) {
        stop("resident_generate(): model is not active; ",
             "call resident_activate() first")
    }
    if (isTRUE(res$in_flight)) {
        stop("resident_generate(): another generation is in flight")
    }
    res$in_flight <- TRUE
    on.exit(res$in_flight <- FALSE, add = TRUE)
    generate(res$model, text, voice, ...)
}

#' Status of a Resident Model
#'
#' Callable in every state, including "broken" and "unloaded".
#'
#' @param res A `chatterbox_resident` handle.
#' @return A list: variant, state, in_flight, device (the exact bound
#'   device, e.g. `"cuda:0"`), pinned_bytes, gpu_bytes (logical sums
#'   over the tensor manifest, not allocator statistics; in the "broken"
#'   state this is the observed on-GPU manifest bytes, or NA when
#'   unknowable), identity (per-artifact sha256, HF repo/revision,
#'   variant), paths (diagnostic), last_error.
#' @export
resident_status <- function(res) {
    if (!inherits(res, "chatterbox_resident")) {
        stop("not a chatterbox_resident handle")
    }
    list(
        # `model` and `dtype` mirror whisper's resident_status() so a host
        # consuming both packages can read one set of field names; `variant`
        # and `paths` are the chatterbox-specific extras.
        model = res$identity$variant,
        variant = res$identity$variant,
        state = res$state,
        in_flight = isTRUE(res$in_flight),
        device = if (is.null(res$target_device)) NA_character_ else
            as.character(res$target_device),
        dtype = res$identity$dtype,
        pinned_bytes = if (identical(res$state, "unloaded")) 0 else
            res$pinned_bytes,
        gpu_bytes = res$gpu_bytes,
        identity = res$identity,
        paths = res$paths,
        last_error = res$last_error
    )
}

#' Unload a Resident Model (Drop Pinned and GPU State)
#'
#' Releases both representations: any GPU tensors and the pinned host
#' copies. Permitted from every state except mid-flight -- including
#' "broken", where it is the recovery path. The handle becomes
#' "unloaded", which is terminal.
#'
#' @param res A `chatterbox_resident` handle.
#' @return The handle, invisibly.
#' @export
resident_unload <- function(res) {
    if (!inherits(res, "chatterbox_resident")) {
        stop("not a chatterbox_resident handle")
    }
    if (isTRUE(res$in_flight)) {
        stop("resident_unload(): a generation is in flight")
    }
    if (identical(res$state, "unloaded")) {
        return(invisible(res))
    }
    if (identical(res$state, "active")) {
        try(.resident_rollback(res), silent = TRUE)
    }
    res$pinned <- NULL
    res$model <- NULL
    res$gpu_bytes <- 0
    res$pinned_bytes <- 0
    res$state <- "unloaded"
    gc()
    try(torch::cuda_empty_cache(), silent = TRUE)
    invisible(res)
}

#' Print a Resident Handle
#'
#' @param x A `chatterbox_resident` handle.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.chatterbox_resident <- function(x, ...) {
    cat(sprintf("<chatterbox_resident: %s [%s] pinned %s>\n",
                if (is.null(x$identity)) "?" else x$identity$variant,
                x$state, .fmt_bytes(x$pinned_bytes)))
    invisible(x)
}
