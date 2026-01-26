#!/usr/bin/env Rscript
# Test CFM Flow module against Python reference

library(torch)

# Source the package
devtools::load_all("/home/troy/chatterbox")

# Load Python reference data
cat("Loading Python reference data...\n")
py_ref <- safetensors::safe_load_file(
    "/home/troy/chatterbox/outputs/cfm_steps.safetensors",
    framework = "torch"
)

cat("Python reference keys:", paste(names(py_ref), collapse = ", "), "\n")

# Check shapes
cat("\nPython shapes:\n")
for (k in names(py_ref)) {
    cat(sprintf("  %s: %s\n", k, paste(py_ref[[k]]$shape, collapse = "x")))
}

# Load S3Gen weights
cat("\nLoading S3Gen weights...\n")
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")
cat("Loaded", length(state_dict), "weight tensors\n")

# Check flow keys
flow_keys <- names(state_dict)[startsWith(names(state_dict), "flow.")]
cat("\nFlow weight keys (first 20):\n")
cat(head(flow_keys, 20), sep = "\n")
cat("...\nTotal flow keys:", length(flow_keys), "\n")

# Create flow module
cat("\nCreating CausalMaskedDiffWithXvec...\n")
flow <- causal_masked_diff_xvec()

# Load weights
cat("\nLoading flow weights...\n")
torch::with_no_grad({
        # Input embedding
        flow$input_embedding$weight$copy_(state_dict[["flow.input_embedding.weight"]])

        # Speaker embedding projection
        flow$spk_embed_affine_layer$weight$copy_(state_dict[["flow.spk_embed_affine_layer.weight"]])
        flow$spk_embed_affine_layer$bias$copy_(state_dict[["flow.spk_embed_affine_layer.bias"]])

        # Encoder projection
        flow$encoder_proj$weight$copy_(state_dict[["flow.encoder_proj.weight"]])
        flow$encoder_proj$bias$copy_(state_dict[["flow.encoder_proj.bias"]])
    })
flow$eval()

# Test 1: Token embedding
cat("\n=== Test 1: Token embedding ===\n")
torch::with_no_grad({
        all_tokens <- py_ref$all_tokens
        cat("all_tokens shape:", paste(all_tokens$shape, collapse = "x"), "\n")

        # Clamp and add 1 for R indexing
        tokens_clamped <- torch::torch_clamp(all_tokens, min = 0L, max = 6560L)$to(dtype = torch::torch_long())
        r_token_emb <- flow$input_embedding$forward(tokens_clamped$add(1L))
        cat("R token_emb shape:", paste(r_token_emb$shape, collapse = "x"), "\n")

        py_token_emb <- py_ref$token_emb
        cat("Python token_emb shape:", paste(py_token_emb$shape, collapse = "x"), "\n")

        diff <- (r_token_emb - py_token_emb)$abs()$max()$item()
        cat("Token embedding max diff:", diff, "\n")
    })

# Test 2: Speaker embedding projection
cat("\n=== Test 2: Speaker embedding projection ===\n")
torch::with_no_grad({
        embedding <- py_ref$embedding
        cat("embedding shape:", paste(embedding$shape, collapse = "x"), "\n")

        emb_norm <- torch::nnf_normalize(embedding, dim = 2L)
        r_spk_emb <- flow$spk_embed_affine_layer$forward(emb_norm)
        cat("R spk_emb shape:", paste(r_spk_emb$shape, collapse = "x"), "\n")

        py_spk_emb <- py_ref$spk_emb
        cat("Python spk_emb shape:", paste(py_spk_emb$shape, collapse = "x"), "\n")

        diff <- (r_spk_emb - py_spk_emb)$abs()$max()$item()
        cat("Speaker embedding max diff:", diff, "\n")
    })

# Test 3: Encoder projection
cat("\n=== Test 3: Encoder projection ===\n")
torch::with_no_grad({
        # Use Python's encoder output directly
        encoder_h <- py_ref$encoder_h
        cat("encoder_h shape:", paste(encoder_h$shape, collapse = "x"), "\n")

        # Trim lookahead (3 tokens * 2 mel ratio = 6 frames)
        h_trimmed <- encoder_h[, 1:(encoder_h$size(2) - 6),]
        cat("h_trimmed shape:", paste(h_trimmed$shape, collapse = "x"), "\n")

        r_h_proj <- flow$encoder_proj$forward(h_trimmed)
        cat("R h_proj shape:", paste(r_h_proj$shape, collapse = "x"), "\n")

        py_h_proj <- py_ref$h_proj
        cat("Python h_proj shape:", paste(py_h_proj$shape, collapse = "x"), "\n")

        diff <- (r_h_proj - py_h_proj)$abs()$max()$item()
        cat("Encoder projection max diff:", diff, "\n")
    })

# Test 4: Check CFM decoder structure
cat("\n=== Test 4: CFM Decoder structure ===\n")
cat("decoder type:", class(flow$decoder)[1], "\n")
cat("decoder.estimator type:", class(flow$decoder$estimator)[1], "\n")

# Check Python decoder keys
decoder_keys <- names(state_dict)[startsWith(names(state_dict), "flow.decoder.")]
cat("\nDecoder weight keys (first 30):\n")
cat(head(decoder_keys, 30), sep = "\n")

cat("\n=== Done ===\n")

