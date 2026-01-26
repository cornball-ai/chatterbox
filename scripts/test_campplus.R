#!/usr/bin/env Rscript
# Test CAMPPlus speaker encoder against Python reference

library(torch)

# Source the package
devtools::load_all("/home/troy/chatterbox")

# Load Python reference data
cat("Loading Python reference data...\n")
py_ref <- read_safetensors(
    "/home/troy/chatterbox/outputs/campplus_reference.safetensors",
    device = "cpu"
)

cat("Python reference keys:", paste(names(py_ref), collapse = ", "), "\n")

# Check shapes
cat("\nPython shapes:\n")
cat("  fbank:", paste(dim(py_ref$fbank), collapse = "x"), "\n")
cat("  after_fcm:", paste(dim(py_ref$after_fcm), collapse = "x"), "\n")
cat("  after_tdnn:", paste(dim(py_ref$after_tdnn), collapse = "x"), "\n")
cat("  embedding:", paste(dim(py_ref$embedding), collapse = "x"), "\n")

# Load S3Gen weights for CAMPPlus
cat("\nLoading S3Gen weights...\n")
weights_path <- "~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"
weights_files <- Sys.glob(path.expand(weights_path))
if (length(weights_files) == 0) {
    stop("Could not find s3gen.safetensors")
}
weights_path <- weights_files[1]
cat("Using weights:", weights_path, "\n")

state_dict <- read_safetensors(weights_path, device = "cpu")
cat("Loaded", length(state_dict), "weight tensors\n")

# Create and load CAMPPlus model
cat("\nCreating CAMPPlus model...\n")
model <- campplus()
torch::with_no_grad({
        load_campplus_weights(model, state_dict, prefix = "speaker_encoder.")
    })
model$eval()

# Run through CAMPPlus step by step using Python's fbank as input
cat("\n=== Testing CAMPPlus with Python fbank ===\n")

py_fbank <- py_ref$fbank
cat("Python fbank shape:", paste(dim(py_fbank), collapse = "x"), "\n")
cat("Python fbank mean:", py_fbank$mean()$item(), "std:", py_fbank$std()$item(), "\n")

torch::with_no_grad({
        # Step 1: Permute (B, T, F) -> (B, F, T)
        x <- py_fbank$permute(c(1L, 3L, 2L))
        cat("\n1. After permute:", paste(dim(x), collapse = "x"), "\n")

        diff <- (x - py_ref$after_permute)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff, scientific = FALSE), "\n")

        # Step 2: FCM head
        x_fcm <- model$head$forward(x)
        cat("\n2. After FCM head:", paste(dim(x_fcm), collapse = "x"), "\n")

        diff_fcm <- (x_fcm - py_ref$after_fcm)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_fcm, scientific = FALSE), "\n")
        cat("   R mean:", x_fcm$mean()$item(), "std:", x_fcm$std()$item(), "\n")
        cat("   Py mean:", py_ref$after_fcm$mean()$item(), "std:", py_ref$after_fcm$std()$item(), "\n")

        # Step 3: TDNN
        x_tdnn <- model$tdnn$forward(x_fcm)
        cat("\n3. After TDNN:", paste(dim(x_tdnn), collapse = "x"), "\n")

        diff_tdnn <- (x_tdnn - py_ref$after_tdnn)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_tdnn, scientific = FALSE), "\n")

        # Step 4: Block 1 + transit
        x_block1 <- model$block1$forward(x_tdnn)
        cat("\n4. After block1:", paste(dim(x_block1), collapse = "x"), "\n")

        diff_block1 <- (x_block1 - py_ref$after_block1)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_block1, scientific = FALSE), "\n")

        x_transit1 <- model$transit1$forward(x_block1)
        cat("\n5. After transit1:", paste(dim(x_transit1), collapse = "x"), "\n")

        diff_transit1 <- (x_transit1 - py_ref$after_transit1)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_transit1, scientific = FALSE), "\n")

        # Step 5: Block 2 + transit
        x_block2 <- model$block2$forward(x_transit1)
        cat("\n6. After block2:", paste(dim(x_block2), collapse = "x"), "\n")

        diff_block2 <- (x_block2 - py_ref$after_block2)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_block2, scientific = FALSE), "\n")

        x_transit2 <- model$transit2$forward(x_block2)
        cat("\n7. After transit2:", paste(dim(x_transit2), collapse = "x"), "\n")

        diff_transit2 <- (x_transit2 - py_ref$after_transit2)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_transit2, scientific = FALSE), "\n")

        # Step 6: Block 3 + transit
        x_block3 <- model$block3$forward(x_transit2)
        cat("\n8. After block3:", paste(dim(x_block3), collapse = "x"), "\n")

        diff_block3 <- (x_block3 - py_ref$after_block3)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_block3, scientific = FALSE), "\n")

        x_transit3 <- model$transit3$forward(x_block3)
        cat("\n9. After transit3:", paste(dim(x_transit3), collapse = "x"), "\n")

        diff_transit3 <- (x_transit3 - py_ref$after_transit3)$abs()$max()$item()
        cat("   Max diff vs Python:", format(diff_transit3, scientific = FALSE), "\n")

        # Step 7: Output nonlinear
        x_out <- torch::nnf_relu(model$out_bn$forward(x_transit3))
        cat("\n10. After out_nonlinear:", paste(dim(x_out), collapse = "x"), "\n")

        diff_out <- (x_out - py_ref$after_out_nonlinear)$abs()$max()$item()
        cat("    Max diff vs Python:", format(diff_out, scientific = FALSE), "\n")

        # Step 8: Stats pooling
        x_stats <- statistics_pooling(x_out)
        cat("\n11. After stats pooling:", paste(dim(x_stats), collapse = "x"), "\n")

        diff_stats <- (x_stats - py_ref$stats_pooled)$abs()$max()$item()
        cat("    Max diff vs Python:", format(diff_stats, scientific = FALSE), "\n")

        # Step 9: Dense layer
        embedding <- model$dense$forward(x_stats)
        cat("\n12. Final embedding:", paste(dim(embedding), collapse = "x"), "\n")

        diff_emb <- (embedding - py_ref$embedding)$abs()$max()$item()
        cat("    Max diff vs Python:", format(diff_emb, scientific = FALSE), "\n")

        # Full forward pass
        full_emb <- model$forward(py_fbank)
        diff_full <- (full_emb - py_ref$full_embedding)$abs()$max()$item()
        cat("\n13. Full forward max diff:", format(diff_full, scientific = FALSE), "\n")
    })

# Summary
cat("\n=== Summary ===\n")
results <- list(
    list("FCM head", diff_fcm, 0.001),
    list("TDNN", diff_tdnn, 0.001),
    list("Block1", diff_block1, 0.01),
    list("Transit1", diff_transit1, 0.01),
    list("Block2", diff_block2, 0.01),
    list("Transit2", diff_transit2, 0.01),
    list("Block3", diff_block3, 0.01),
    list("Transit3", diff_transit3, 0.01),
    list("Out nonlinear", diff_out, 0.01),
    list("Stats pooling", diff_stats, 0.01),
    list("Embedding", diff_emb, 0.01),
    list("Full forward", diff_full, 0.01)
)

pass_count <- 0
for (r in results) {
    status <- if (r[[2]] < r[[3]]) { pass_count <- pass_count + 1;"PASS" } else "FAIL"
    cat(sprintf("%-15s: %.6f (tol: %.4f) %s\n", r[[1]], r[[2]], r[[3]], status))
}

cat(sprintf("\nTotal: %d/%d passed\n", pass_count, length(results)))

