#!/usr/bin/env Rscript
# Test UpsampleConformerEncoder against Python reference

library(torch)

# Source the package
devtools::load_all("/home/troy/chatterbox")

# Load Python reference data
cat("Loading Python reference data...\n")
py_ref <- read_safetensors(
    "/home/troy/chatterbox/outputs/conformer_reference.safetensors",
    device = "cpu"
)

cat("Python reference keys:\n")
for (k in names(py_ref)) {
    cat(sprintf("  %s: %s\n", k, paste(dim(py_ref[[k]]), collapse = "x")))
}

# Load S3Gen weights
cat("\nLoading S3Gen weights...\n")
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")
cat("Loaded", length(state_dict), "weight tensors\n")

# Create encoder using our full implementation
cat("\nCreating UpsampleConformerEncoder...\n")
encoder <- upsample_conformer_encoder_full(
    input_size = 512L,
    output_size = 512L,
    num_blocks = 6L,
    num_up_blocks = 4L,
    n_head = 8L,
    n_ffn = 2048L,
    dropout_rate = 0.1,
    pre_lookahead_len = 3L
)

# Load weights
cat("Loading encoder weights...\n")
torch::with_no_grad({
        load_conformer_encoder_weights(encoder, state_dict, prefix = "flow.encoder.")
    })
encoder$eval()

# Get test input from Python reference
x <- py_ref$test_input
x_lens <- py_ref$test_lens$to(dtype = torch::torch_long())
cat("\nTest input x shape:", paste(dim(x), collapse = "x"), "\n")
cat("Test input x_lens:", as.integer(x_lens$cpu()), "\n")

# Run forward pass
cat("\n=== Running forward pass ===\n")
torch::with_no_grad({
        result <- encoder$forward(x, x_lens)
        r_output <- result[[1]]
        r_masks <- result[[2]]

        cat("R output shape:", paste(dim(r_output), collapse = "x"), "\n")
        cat("Python output shape:", paste(dim(py_ref$full_output), collapse = "x"), "\n")

        # Compare output
        diff_output <- (r_output - py_ref$full_output)$abs()$max()$item()
        cat("\nFull output max diff:", diff_output, "\n")

        # Statistics comparison
        cat("\nStatistics comparison:\n")
        cat("  R output - mean:", r_output$mean()$item(), "std:", r_output$std()$item(), "\n")
        cat("  Py output - mean:", py_ref$full_output$mean()$item(), "std:", py_ref$full_output$std()$item(), "\n")
    })

# Compare intermediates if available
cat("\n=== Comparing intermediates ===\n")
torch::with_no_grad({
        T <- x$size(2)
        device <- x$device

        # Create mask
        masks <- encoder$make_pad_mask(x_lens, T, device)$unsqueeze(2)
        masks <- !masks

        # Step 1: embed
        embed_result <- encoder$embed$forward(x, masks)
        r_xs <- embed_result[[1]]
        r_pos_emb <- embed_result[[2]]

        if ("after_embed" %in% names(py_ref)) {
            diff_embed <- (r_xs - py_ref$after_embed)$abs()$max()$item()
            cat("After embed max diff:", diff_embed, "\n")
        }

        # Step 2: pre_lookahead
        r_pll <- encoder$pre_lookahead_layer$forward(r_xs)

        if ("after_pre_lookahead" %in% names(py_ref)) {
            diff_pll <- (r_pll - py_ref$after_pre_lookahead)$abs()$max()$item()
            cat("After pre_lookahead max diff:", diff_pll, "\n")
        }

        # Step 3: encoders
        xs_enc <- r_pll
        mask_pad <- masks
        for (i in seq_along(encoder$encoders)) {
            result <- encoder$encoders[[i]]$forward(xs_enc, masks, r_pos_emb, mask_pad)
            xs_enc <- result[[1]]
        }

        if ("after_encoders" %in% names(py_ref)) {
            diff_enc <- (xs_enc - py_ref$after_encoders)$abs()$max()$item()
            cat("After encoders max diff:", diff_enc, "\n")
        }

        # Step 4: transpose + upsample
        xs_t <- xs_enc$transpose(2L, 3L)$contiguous()
        up_result <- encoder$up_layer$forward(xs_t, x_lens)
        xs_up <- up_result[[1]]
        xs_up_lens <- up_result[[2]]

        if ("after_upsample" %in% names(py_ref)) {
            diff_up <- (xs_up - py_ref$after_upsample)$abs()$max()$item()
            cat("After upsample max diff:", diff_up, "\n")
        }

        # Step 5: transpose back + up_embed
        xs_up_t <- xs_up$transpose(2L, 3L)$contiguous()
        T_up <- xs_up_t$size(2)
        masks_up <- encoder$make_pad_mask(xs_up_lens, T_up, device)$unsqueeze(2)
        masks_up <- !masks_up

        up_embed_result <- encoder$up_embed$forward(xs_up_t, masks_up)
        r_xs_up2 <- up_embed_result[[1]]
        r_pos_emb_up <- up_embed_result[[2]]

        if ("after_up_embed" %in% names(py_ref)) {
            diff_up_embed <- (r_xs_up2 - py_ref$after_up_embed)$abs()$max()$item()
            cat("After up_embed max diff:", diff_up_embed, "\n")
        }

        # Step 6: up_encoders
        xs_up_enc <- r_xs_up2
        mask_pad_up <- masks_up
        for (i in seq_along(encoder$up_encoders)) {
            result <- encoder$up_encoders[[i]]$forward(xs_up_enc, masks_up, r_pos_emb_up, mask_pad_up)
            xs_up_enc <- result[[1]]
        }

        if ("after_up_encoders" %in% names(py_ref)) {
            diff_up_enc <- (xs_up_enc - py_ref$after_up_encoders)$abs()$max()$item()
            cat("After up_encoders max diff:", diff_up_enc, "\n")
        }

        # Step 7: final norm
        xs_final <- encoder$after_norm$forward(xs_up_enc)

        if ("after_final_norm" %in% names(py_ref)) {
            diff_final <- (xs_final - py_ref$after_final_norm)$abs()$max()$item()
            cat("After final_norm max diff:", diff_final, "\n")
        }
    })

# Summary
cat("\n=== Summary ===\n")
if (diff_output < 0.01) {
    cat("PASS: Conformer encoder output matches within 0.01\n")
} else if (diff_output < 0.1) {
    cat("WARN: Conformer encoder output differs by", diff_output, "\n")
    cat("This may indicate weight loading issues or architecture differences\n")
} else {
    cat("FAIL: Conformer encoder output differs significantly:", diff_output, "\n")
}

cat("\n=== Done ===\n")

