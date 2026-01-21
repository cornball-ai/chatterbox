#!/usr/bin/env Rscript
# Debug Llama weight loading

rhydrogen::load_all()

cat("=== Loading T3 Weights ===\n")
paths <- download_chatterbox_models()
t3_weights <- read_safetensors(paths$t3_cfg, device = "cpu")

# Check which keys are in the weights
cat("\nAll weight keys (first 50):\n")
all_keys <- names(t3_weights)
for (key in all_keys[1:min(50, length(all_keys))]) {
  tensor <- t3_weights[[key]]
  cat(sprintf("  %s: %s, mean=%.6f, std=%.6f\n",
              key, paste(dim(tensor), collapse = "x"),
              tensor$mean()$item(), tensor$std()$item()))
}

cat("\n\nLlama backbone keys (tfmr.*):\n")
tfmr_keys <- grep("^tfmr\\.", all_keys, value = TRUE)
cat("Total tfmr keys:", length(tfmr_keys), "\n")
for (key in tfmr_keys[1:min(20, length(tfmr_keys))]) {
  tensor <- t3_weights[[key]]
  cat(sprintf("  %s: %s, mean=%.6f\n", key, paste(dim(tensor), collapse = "x"), tensor$mean()$item()))
}

# Create the model and check weights
cat("\n=== Creating T3 Model ===\n")
model <- t3_model()

# Check model parameters before loading
cat("\nModel parameters (before loading):\n")
cat("text_emb weight mean:", model$text_emb$weight$mean()$item(), "\n")
cat("speech_emb weight mean:", model$speech_emb$weight$mean()$item(), "\n")
cat("tfmr layer 0 input_layernorm weight mean:", model$tfmr$layers[[1]]$input_layernorm$weight$mean()$item(), "\n")

# Load weights
cat("\n=== Loading Weights ===\n")
model <- load_t3_weights(model, t3_weights)

# Check model parameters after loading
cat("\nModel parameters (after loading):\n")
cat("text_emb weight mean:", model$text_emb$weight$mean()$item(), "\n")
cat("speech_emb weight mean:", model$speech_emb$weight$mean()$item(), "\n")
cat("tfmr layer 0 input_layernorm weight mean:", model$tfmr$layers[[1]]$input_layernorm$weight$mean()$item(), "\n")
cat("tfmr layer 0 q_proj weight mean:", model$tfmr$layers[[1]]$self_attn$q_proj$weight$mean()$item(), "\n")
cat("tfmr layer 0 mlp gate_proj weight mean:", model$tfmr$layers[[1]]$mlp$gate_proj$weight$mean()$item(), "\n")

# Compare with file weights
cat("\n=== Comparing with file weights ===\n")
file_emb <- t3_weights[["text_emb.weight"]]
model_emb <- model$text_emb$weight
cat("text_emb - file mean:", file_emb$mean()$item(), ", model mean:", model_emb$mean()$item(), "\n")

file_ln <- t3_weights[["tfmr.layers.0.input_layernorm.weight"]]
model_ln <- model$tfmr$layers[[1]]$input_layernorm$weight
cat("layer0 ln - file mean:", file_ln$mean()$item(), ", model mean:", model_ln$mean()$item(), "\n")

# Test forward pass with simple input
cat("\n=== Test Forward Pass ===\n")
device <- "cuda"
model$to(device = device)
model$eval()

# Create simple input
test_input <- torch::torch_randn(c(1, 10, 1024), device = device)
cat("Test input mean:", test_input$mean()$item(), "\n")

torch::with_no_grad({
  output <- model$tfmr$forward(inputs_embeds = test_input, use_cache = FALSE)
  cat("Output last_hidden_state shape:", paste(dim(output$last_hidden_state), collapse = "x"), "\n")
  cat("Output last_hidden_state mean:", output$last_hidden_state$mean()$item(), "\n")
  cat("Output last_hidden_state std:", output$last_hidden_state$std()$item(), "\n")
  cat("Output range: [", output$last_hidden_state$min()$item(), ", ", output$last_hidden_state$max()$item(), "]\n", sep = "")
})
