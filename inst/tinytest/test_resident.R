# Tests for in-process model residency (R/resident.R).
#
# CPU-safe section runs everywhere. The CUDA section needs a GPU and the
# downloaded models, so it is gated on cuda_is_available() && at_home().

if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}

# ---- byte formatting ----
fmt <- chatterbox:::.fmt_bytes
expect_equal(fmt(0), "0 B")
expect_equal(fmt(1024), "1.00 KB")
expect_equal(fmt(4.1 * 1024^3), "4.10 GB")

# ---- closed-behavior matrix on fabricated handles (no GPU needed) ----

fab <- function(state, in_flight = FALSE) {
    res <- new.env(parent = emptyenv())
    res$state <- state
    res$in_flight <- in_flight
    res$identity <- list(variant = "fake")
    res$pinned_bytes <- 0
    res$gpu_bytes <- 0
    res$last_error <- "synthetic"
    class(res) <- "chatterbox_resident"
    res
}

u <- fab("unloaded")
expect_error(resident_activate(u), pattern = "unloaded")
expect_error(resident_deactivate(u), pattern = "unloaded")
expect_error(resident_generate(u, "x", "v.wav"), pattern = "unloaded")
expect_equal(resident_status(u)$state, "unloaded")
expect_silent(resident_unload(u))
expect_equal(u$state, "unloaded")

b <- fab("broken")
expect_error(resident_activate(b), pattern = "broken")
expect_error(resident_deactivate(b), pattern = "broken")
expect_error(resident_generate(b, "x", "v.wav"), pattern = "broken")
expect_equal(resident_status(b)$state, "broken")
expect_equal(resident_status(b)$last_error, "synthetic")
expect_silent(resident_unload(b))
expect_equal(b$state, "unloaded")

i <- fab("inactive")
expect_error(resident_generate(i, "x", "v.wav"),
             pattern = "resident_activate")

a <- fab("active", in_flight = TRUE)
expect_error(resident_deactivate(a), pattern = "in flight")
expect_error(resident_unload(a), pattern = "in flight")
expect_error(resident_generate(a, "x", "v.wav"), pattern = "in flight")

expect_error(resident_status(list()), pattern = "chatterbox_resident")
expect_error(resident_activate(list()), pattern = "chatterbox_resident")

# ---- manifest machinery on a fabricated model shape (CPU torch) ----
# .resident_modules reads $voice_encoder/$t3/$s3gen; NULLs are skipped, so
# a minimal fake with one module exercises the flat-name plumbing.

fake_model <- list(voice_encoder = NULL, t3 = torch::nn_linear(4, 4),
                   s3gen = NULL)
man <- chatterbox:::.resident_manifest(fake_model)
expect_equal(sort(man$name), c("t3.bias", "t3.weight"))
expect_true(all(man$kind == "parameter"))
expect_equal(man$bytes[man$name == "t3.weight"], 4 * 4 * 4)

# buffers ride along, module-prefixed
fake_bn <- list(voice_encoder = NULL, t3 = torch::nn_batch_norm1d(4),
                s3gen = NULL)
manb <- chatterbox:::.resident_manifest(fake_bn)
expect_true("t3.running_mean" %in% manb$name[manb$kind == "buffer"])

# two modules: names stay disjoint via prefixes
fake2 <- list(voice_encoder = torch::nn_linear(2, 2),
              t3 = torch::nn_linear(2, 2), s3gen = NULL)
man2 <- chatterbox:::.resident_manifest(fake2)
expect_equal(sort(man2$name), c("t3.bias", "t3.weight",
                                "voice_encoder.bias",
                                "voice_encoder.weight"))

# exact-equality check: pass, then fail on a mutated manifest
expect_silent(chatterbox:::.resident_check(fake_model, man))
man_bad <- man
man_bad$name[1] <- "t3.nonexistent"
expect_error(chatterbox:::.resident_check(fake_model, man_bad),
             pattern = "manifest")
man_bad2 <- man
man_bad2$shape[man_bad2$name == "t3.weight"] <- "9x9"
expect_error(chatterbox:::.resident_check(fake_model, man_bad2),
             pattern = "shape")

# ---- CUDA section ----

if (!torch::cuda_is_available()) exit_file("CUDA not available")
if (!at_home()) exit_file("CUDA residency tests only run at home")
have_std <- tryCatch({
    chatterbox:::get_model_paths()
    TRUE
}, error = function(e) FALSE)
if (!have_std) exit_file("standard chatterbox models not downloaded")

dev <- torch::torch_device("cuda")
voice <- system.file("audio", "jfk.wav", package = "chatterbox")
alloc <- function() {
    torch::cuda_memory_stats()$allocated_bytes$all$current
}

res <- resident_load(verbose = FALSE)

# name coverage: pinned == manifest == current tensors across all modules
cur <- chatterbox:::.resident_tensors(res$model)
expect_true(setequal(names(res$pinned), res$manifest$name))
expect_true(setequal(names(cur), res$manifest$name))
expect_true(any(grepl("^voice_encoder\\.", res$manifest$name)))
expect_true(any(grepl("^t3\\.", res$manifest$name)))
expect_true(any(grepl("^s3gen\\.", res$manifest$name)))
expect_true(all(vapply(names(res$pinned), function(nm) {
    res$pinned[[nm]]$is_pinned(dev)
}, logical(1))))

# byte counts are logical sums over the manifest
expect_equal(res$pinned_bytes, sum(res$manifest$bytes))
expect_equal(resident_status(res)$gpu_bytes, 0)

# identity: per-artifact digests + resolved repo/revision
st <- resident_status(res)
expect_equal(st$variant, "standard")
expect_equal(st$identity$repo, "ResembleAI/chatterbox")
expect_false(is.na(st$identity$revision))
expect_true(grepl("^[0-9a-f]{40}$", st$identity$revision))
expect_true(length(st$identity$artifacts) >= 3)
expect_true(all(vapply(st$identity$artifacts,
                       function(a) nchar(a$sha256) == 64, logical(1))))
expect_true(grepl("^cuda:[0-9]+$", st$device))

# ---- full cycle: tensor-level equivalence + generation smoke ----
# Full-generation equality is NOT asserted. The standard model samples
# multinomially (top_k gates only the turbo path), and both the CFM
# solver and the vocoder draw noise, so two runs legitimately differ in
# length. The cycle invariants that ARE contractual and are asserted
# here: exact copy fidelity of reactivated weights, and equivalence of
# the deterministic conditioning forward (voice encoder + S3Gen
# reference), run under a fixed seed so any incidental draw is pinned.
base_vram <- alloc()
det <- list(voice, top_k = 1L, temperature = 0.2, backend = "jit",
            max_new_tokens = 120L)

# deterministic intermediate: voice conditioning under a fixed seed
embed <- function(r) {
    torch::with_torch_manual_seed(
        create_voice_embedding(r$model, voice), seed = 1L)
}

# generic tensor walk over a voice-embedding object
collect_tensors <- function(x) {
    if (inherits(x, "torch_tensor")) return(list(x))
    if (is.list(x)) {
        return(unlist(lapply(x, collect_tensors), recursive = FALSE))
    }
    list()
}
emb_diff <- function(a, b) {
    ta <- collect_tensors(a)
    tb <- collect_tensors(b)
    expect_true(length(ta) > 0)
    expect_equal(length(ta), length(tb))
    max(vapply(seq_along(ta), function(i) {
        (ta[[i]]$cpu()$to(dtype = torch::torch_float()) -
         tb[[i]]$cpu()$to(dtype = torch::torch_float()))$abs()$max()$item()
    }, numeric(1)))
}

# model$device must track where the weights actually are: generate() and
# create_voice_embedding() place their inputs by it, so a stale value
# mixes CPU inputs with CUDA weights (or the reverse).
expect_equal(res$model$device, "cpu")

resident_activate(res)
expect_equal(res$state, "active")
expect_equal(resident_status(res)$gpu_bytes, res$pinned_bytes)
expect_equal(res$model$device, as.character(res$target_device))

r1 <- do.call(resident_generate,
              c(list(res, "The quick brown fox jumps over the lazy dog."),
                det))
expect_true(length(r1$audio) > 0)
emb1 <- embed(res)

# copy-fidelity tensor check: a large reactivated CUDA tensor is exactly
# its pinned host copy (no compute involved, so exact equality holds)
big <- res$manifest$name[which.max(res$manifest$bytes)]
cur <- chatterbox:::.resident_tensors(res$model)
expect_equal((cur[[big]]$cpu()$to(dtype = torch::torch_float()) -
              res$pinned[[big]]$to(dtype = torch::torch_float()))$abs()$max()$item(), 0)

resident_deactivate(res)
expect_equal(res$state, "inactive")
expect_equal(resident_status(res)$gpu_bytes, 0)
expect_equal(res$model$device, "cpu")
# VRAM reclaimed (allocator evidence, never part of gpu_bytes)
expect_true(alloc() <= base_vram + 64 * 1024^2)

resident_activate(res)
# module-forward equivalence across the cycle: voice conditioning runs
# the voice_encoder and s3gen reference paths on the reactivated weights
emb2 <- embed(res)
expect_true(emb_diff(emb1, emb2) < 1e-3)
r2 <- do.call(resident_generate,
              c(list(res, "The quick brown fox jumps over the lazy dog."),
                det))
expect_true(length(r2$audio) > 0)

# ---- release = FALSE frees the weights but keeps the allocator pool ----
# Both modes must zero gpu_bytes and return every tensor to pinned host
# memory; they differ only in whether the blocks go back to the driver.
# Retaining them is what makes switching fast in a single-process host.
resident_activate(res)
resident_deactivate(res, release = FALSE)
expect_equal(res$state, "inactive")
expect_equal(resident_status(res)$gpu_bytes, 0)
expect_equal(res$model$device, "cpu")
expect_true(chatterbox:::.resident_verify_pinned(res))
# allocated (per-tensor) drops; reserved (pooled) is allowed to stay
expect_true(alloc() <= base_vram + 64 * 1024^2)
expect_true(torch::cuda_memory_stats()$reserved_bytes$all$current >=
            torch::cuda_memory_stats()$allocated_bytes$all$current)
# and the handle still conditions equivalently afterwards
resident_activate(res)
expect_true(emb_diff(emb1, embed(res)) < 1e-3)
resident_deactivate(res)

# ---- injected mid-activation failure: rollback proven ----
resident_deactivate(res)
expect_equal(res$state, "inactive")

real_move <- res$move
res$move <- function(res, device) {
    ts <- chatterbox:::.resident_tensors(res$model)
    nms <- names(ts)[seq_len(5)]
    torch::with_no_grad({
        for (nm in nms) ts[[nm]]$set_data(ts[[nm]]$to(device = device))
    })
    stop("injected mid-move failure")
}
expect_error(resident_activate(res), pattern = "rolled back")
expect_equal(res$state, "inactive")
expect_true(chatterbox:::.resident_verify_pinned(res))
# a proven rollback also restores the device string
expect_equal(res$model$device, "cpu")
expect_true(alloc() <= base_vram + 64 * 1024^2)
res$move <- real_move

# recovery: the handle still activates, conditions equivalently, and speaks
resident_activate(res)
emb3 <- embed(res)
expect_true(emb_diff(emb1, emb3) < 1e-3)
r3 <- do.call(resident_generate,
              c(list(res, "The quick brown fox jumps over the lazy dog."),
                det))
expect_true(length(r3$audio) > 0)
resident_deactivate(res)

# ---- two resident handles alternating on one GPU (standard + turbo) ----
have_turbo <- tryCatch({
    chatterbox:::get_turbo_model_paths()
    TRUE
}, error = function(e) FALSE)
if (have_turbo) {
    res_t <- resident_load(turbo = TRUE, verbose = FALSE)
    expect_equal(resident_status(res_t)$variant, "turbo")
    for (round in 1:2) {
        resident_activate(res)
        ra <- do.call(resident_generate, c(list(res, "Round trip."), det))
        expect_true(length(ra$audio) > 0)
        resident_deactivate(res)

        resident_activate(res_t)
        rt <- resident_generate(res_t, "Round trip.", voice, top_k = 1L,
                                max_new_tokens = 120L)
        expect_true(length(rt$audio) > 0)
        resident_deactivate(res_t)
    }
    resident_unload(res_t)
    expect_equal(res_t$state, "unloaded")
}

# ---- unprovable rollback -> broken, with observed partial gpu_bytes ----
expect_equal(res$state, "inactive")
res$pinned[[res$manifest$name[1]]] <- NULL
res$move <- function(res, device) {
    ts <- chatterbox:::.resident_tensors(res$model)
    nms <- names(ts)[seq_len(5)]
    torch::with_no_grad({
        for (nm in nms) ts[[nm]]$set_data(ts[[nm]]$to(device = device))
    })
    stop("injected failure, corrupted pinned")
}
expect_error(resident_activate(res), pattern = "broken")
expect_equal(res$state, "broken")
expect_true(grepl("rollback:", resident_status(res)$last_error))
gb_broken <- resident_status(res)$gpu_bytes
expect_true(gb_broken > 0 && gb_broken < res$pinned_bytes)
expect_error(resident_generate(res, "x", voice), pattern = "broken")
expect_error(resident_deactivate(res), pattern = "broken")

resident_unload(res)
expect_equal(res$state, "unloaded")
expect_error(resident_activate(res), pattern = "unloaded")
expect_equal(resident_status(res)$pinned_bytes, 0)
expect_true(alloc() <= base_vram + 64 * 1024^2)
