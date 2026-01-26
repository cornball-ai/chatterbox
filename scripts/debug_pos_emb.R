#!/usr/bin/env r
# Debug position embedding mismatch

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/llama.R")
source("~/chatterbox/R/t3.R")

cat("Loading reference and weights...\n")
ref <- read_safetensors("~/chatterbox/outputs/t3_steps.safetensors")
weights <- read_safetensors("~/.cache/chatterbox/ResembleAI--chatterbox/t3_cfg.safetensors")

# Create model and load weights
config <- t3_config_english()
model <- t3_model(config)
model <- load_t3_weights(model, weights)
model$eval()

py_text_tokens <- ref$text_tokens
py_text_emb <- ref$text_emb

cat("\n=== Text Tokens ===\n")
cat(sprintf("Text tokens: %s\n", paste(as.integer(py_text_tokens), collapse = ", ")))

cat("\n=== R Position Embedding Weights ===\n")
cat(sprintf("text_pos_emb weight shape: %s\n",
        paste(dim(model$text_pos_emb$emb$weight), collapse = "x")))
cat(sprintf("First 5 position vectors (rows 1-5):\n"))
for (i in 1:5) {
    vec <- model$text_pos_emb$emb$weight[i, 1:5]
    cat(sprintf("  Row %d: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
            i, as.numeric(vec)[1], as.numeric(vec)[2], as.numeric(vec)[3],
            as.numeric(vec)[4], as.numeric(vec)[5]))
}

cat("\n=== Python Position Embedding Weights (from saved file) ===\n")
py_pos_weight <- weights[["text_pos_emb.emb.weight"]]
cat(sprintf("Python weight shape: %s\n", paste(dim(py_pos_weight), collapse = "x")))
for (i in 1:5) {
    vec <- py_pos_weight[i, 1:5]
    cat(sprintf("  Row %d: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
            i, as.numeric(vec)[1], as.numeric(vec)[2], as.numeric(vec)[3],
            as.numeric(vec)[4], as.numeric(vec)[5]))
}

cat("\n=== Position Embedding Forward Test ===\n")
# Test getting position embeddings for sequence length 8
# Create dummy input with shape [1, 8]
dummy <- torch::torch_zeros(c(1L, 8L))

# R position embedding forward
torch::with_no_grad({
        r_pos_emb <<- model$text_pos_emb$forward(dummy)
    })
cat(sprintf("R pos emb shape: %s\n", paste(dim(r_pos_emb), collapse = "x")))
cat(sprintf("R pos emb first token (pos 0): [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
        as.numeric(r_pos_emb[1, 1:5])[1], as.numeric(r_pos_emb[1, 1:5])[2],
        as.numeric(r_pos_emb[1, 1:5])[3], as.numeric(r_pos_emb[1, 1:5])[4],
        as.numeric(r_pos_emb[1, 1:5])[5]))
cat(sprintf("R pos emb second token (pos 1): [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
        as.numeric(r_pos_emb[2, 1:5])[1], as.numeric(r_pos_emb[2, 1:5])[2],
        as.numeric(r_pos_emb[2, 1:5])[3], as.numeric(r_pos_emb[2, 1:5])[4],
        as.numeric(r_pos_emb[2, 1:5])[5]))

# Compare token embeddings (without position) - this should match
cat("\n=== Token Embedding (No Position) ===\n")
torch::with_no_grad({
        r_tok_emb <<- model$text_emb$forward(py_text_tokens$add(1L))
    })
cat(sprintf("R token emb shape: %s\n", paste(dim(r_tok_emb), collapse = "x")))

# Python text_emb already includes position - subtract token emb to get position
py_pos_only <- py_text_emb - r_tok_emb
cat("Py position component (text_emb - token_emb):\n")
vals <- as.numeric(py_pos_only[1, 1, 1:5])
cat(sprintf("  First token: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
        vals[1], vals[2], vals[3], vals[4], vals[5]))

cat("\n=== Weight Row Comparison ===\n")
# Note: R tensor indexing is 1-based, so [1,] accesses internal row 0
cat("Weight row 1 (internal row 0, Python position 0): ")
vals <- as.numeric(py_pos_weight[1, 1:3])
cat(sprintf("[%.4f, %.4f, %.4f]\n", vals[1], vals[2], vals[3]))

cat("Weight row 2 (internal row 1, Python position 1): ")
vals <- as.numeric(py_pos_weight[2, 1:3])
cat(sprintf("[%.4f, %.4f, %.4f]\n", vals[1], vals[2], vals[3]))

cat("\nPy position component first token (from saved ref): ")
vals <- as.numeric(py_pos_only[1, 1, 1:3])
cat(sprintf("[%.4f, %.4f, %.4f]\n", vals[1], vals[2], vals[3]))

cat("\nR pos emb first token (from forward): ")
vals <- as.numeric(r_pos_emb[1, 1:3])
cat(sprintf("[%.4f, %.4f, %.4f]\n", vals[1], vals[2], vals[3]))

cat("\nDone.\n")

