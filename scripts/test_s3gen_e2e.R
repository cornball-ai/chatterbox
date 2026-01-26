#!/usr/bin/env Rscript
# Test S3Gen end-to-end

library(torch)

# Source the package
devtools::load_all("/home/troy/chatterbox")

# Load S3Gen components reference
cat("Loading Python reference data...\n")
py_ref <- safetensors::safe_load_file(
    "/home/troy/chatterbox/outputs/s3gen_components.safetensors",
    framework = "torch"
)

cat("Python reference keys:", paste(names(py_ref), collapse = ", "), "\n")
cat("\nPython output_wav shape:", paste(py_ref$output_wav$shape, collapse = "x"), "\n")
cat("Python output_wav mean:", py_ref$output_wav$mean()$item(), "\n")
cat("Python output_wav std:", py_ref$output_wav$std()$item(), "\n")

# Load weights
cat("\nLoading S3Gen weights...\n")
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")

# Check what modules exist in s3gen.R
cat("\nCreating S3Gen model...\n")
tryCatch({
        model <- s3gen()
        cat("S3Gen model created\n")
        cat("Model components:\n")
        cat("  tokenizer:", class(model$tokenizer)[1], "\n")
        cat("  speaker_encoder:", class(model$speaker_encoder)[1], "\n")
        cat("  flow:", class(model$flow)[1], "\n")
        cat("  mel2wav:", if (is.null(model$mel2wav)) "NULL" else class(model$mel2wav)[1], "\n")
    }, error = function (e)
    {
        cat("Error creating S3Gen:", e$message, "\n")
    })

# Try running with reference data
cat("\n=== Testing flow.forward with Python inputs ===\n")
tryCatch({
        torch::with_no_grad({
                # Load flow weights
                model$flow$input_embedding$weight$copy_(state_dict[["flow.input_embedding.weight"]])
                model$flow$spk_embed_affine_layer$weight$copy_(state_dict[["flow.spk_embed_affine_layer.weight"]])
                model$flow$spk_embed_affine_layer$bias$copy_(state_dict[["flow.spk_embed_affine_layer.bias"]])
                model$flow$encoder_proj$weight$copy_(state_dict[["flow.encoder_proj.weight"]])
                model$flow$encoder_proj$bias$copy_(state_dict[["flow.encoder_proj.bias"]])
            })
        model$eval()

        # Build ref_dict from Python reference
        ref_dict <- list(
            prompt_token = py_ref$prompt_token,
            prompt_token_len = py_ref$prompt_token_len,
            prompt_feat = py_ref$prompt_feat,
            prompt_feat_len = NULL,
            embedding = py_ref$embedding
        )

        cat("ref_dict created:\n")
        cat("  prompt_token:", paste(ref_dict$prompt_token$shape, collapse = "x"), "\n")
        cat("  prompt_feat:", paste(ref_dict$prompt_feat$shape, collapse = "x"), "\n")
        cat("  embedding:", paste(ref_dict$embedding$shape, collapse = "x"), "\n")

        # Run flow.forward
        cat("\nRunning flow$forward...\n")
        torch::with_no_grad({
                result <- model$flow$forward(
                    token = py_ref$test_tokens,
                    token_len = torch::torch_tensor(py_ref$test_tokens$size(2)),
                    prompt_token = ref_dict$prompt_token,
                    prompt_token_len = ref_dict$prompt_token_len,
                    prompt_feat = ref_dict$prompt_feat,
                    prompt_feat_len = ref_dict$prompt_feat_len,
                    embedding = ref_dict$embedding,
                    finalize = FALSE
                )
            })

        output_mels <- result[[1]]
        cat("R output_mels shape:", paste(output_mels$shape, collapse = "x"), "\n")
        cat("R output_mels mean:", output_mels$mean()$item(), "\n")
        cat("R output_mels std:", output_mels$std()$item(), "\n")

        # Compare with Python
        py_mels <- py_ref$output_mels
        cat("\nPython output_mels shape:", paste(py_mels$shape, collapse = "x"), "\n")
        cat("Python output_mels mean:", py_mels$mean()$item(), "\n")
        cat("Python output_mels std:", py_mels$std()$item(), "\n")

    }, error = function (e)
    {
        cat("Error:", e$message, "\n")
        print(e)
    })

cat("\n=== Done ===\n")

