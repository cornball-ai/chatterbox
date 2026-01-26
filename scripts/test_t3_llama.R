#!/usr/bin/env r
# Test R T3 Llama backbone against Python reference

library(torch)
source("~/chatterbox/R/safetensors.R")
source("~/chatterbox/R/llama.R")
source("~/chatterbox/R/t3.R")

cat("Loading Python reference...\n")
ref <- read_safetensors("~/chatterbox/outputs/t3_steps.safetensors")
cat("Reference keys:", paste(names(ref), collapse = ", "), "\n\n")

# Load T3 weights
cat("Loading T3 weights...\n")
weights <- read_safetensors("~/.cache/chatterbox/ResembleAI--chatterbox/t3_cfg.safetensors")
cat(sprintf("Loaded %d weight tensors\n\n", length(weights)))

# Reference values
py_input_embeds <- ref$input_embeds
py_hidden_states <- ref$hidden_states
py_speech_logits <- ref$speech_logits

cat("=== Python Reference ===\n")
cat(sprintf("Input embeds: %s\n", paste(dim(py_input_embeds), collapse = "x")))
cat(sprintf("Hidden states: %s\n", paste(dim(py_hidden_states), collapse = "x")))
cat(sprintf("  mean=%.6f, std=%.6f\n",
        py_hidden_states$mean()$item(), py_hidden_states$std()$item()))
cat(sprintf("Speech logits: %s\n", paste(dim(py_speech_logits), collapse = "x")))
cat(sprintf("  mean=%.6f, std=%.6f\n",
        py_speech_logits$mean()$item(), py_speech_logits$std()$item()))

# ============================================================================
# Create and load T3 model
# ============================================================================
cat("\n=== Creating R T3 Model ===\n")
config <- t3_config_english()
model <- t3_model(config)
model <- load_t3_weights(model, weights)
model$eval()
cat("Model loaded\n")

# ============================================================================
# Run Llama forward pass
# ============================================================================
cat("\n=== Llama Forward Pass ===\n")

torch::with_no_grad({
        # Run through Llama transformer
        output <- model$tfmr$forward(
            inputs_embeds = py_input_embeds,
            use_cache = FALSE,
            output_hidden_states = TRUE
        )

        r_hidden_states <- output$last_hidden_state
        cat(sprintf("R hidden states: %s\n", paste(dim(r_hidden_states), collapse = "x")))
        cat(sprintf("  mean=%.6f, std=%.6f\n",
                r_hidden_states$mean()$item(), r_hidden_states$std()$item()))

        # Compare hidden states
        diff_hidden <- (r_hidden_states - py_hidden_states)$abs()$max()$item()
        mean_diff_hidden <- (r_hidden_states - py_hidden_states)$abs()$mean()$item()
        cat(sprintf("Hidden states max diff: %.6f\n", diff_hidden))
        cat(sprintf("Hidden states mean diff: %.6f\n", mean_diff_hidden))

        # Get speech logits
        r_speech_logits <- model$speech_head$forward(r_hidden_states[, 43,, drop = FALSE])
        cat(sprintf("\nR speech logits: %s\n", paste(dim(r_speech_logits), collapse = "x")))
        cat(sprintf("  mean=%.6f, std=%.6f\n",
                r_speech_logits$mean()$item(), r_speech_logits$std()$item()))

        # Compare speech logits
        diff_logits <- (r_speech_logits - py_speech_logits)$abs()$max()$item()
        mean_diff_logits <- (r_speech_logits - py_speech_logits)$abs()$mean()$item()
        cat(sprintf("Speech logits max diff: %.6f\n", diff_logits))
        cat(sprintf("Speech logits mean diff: %.6f\n", mean_diff_logits))
    })

# ============================================================================
# Position-by-position comparison
# ============================================================================
cat("\n=== Position-by-Position Hidden States ===\n")

# Check first few positions
torch::with_no_grad({
        for (pos in c(1, 17, 34, 43)) { # Conditioning, middle, end
            r_pos <- r_hidden_states[, pos,]
            py_pos <- py_hidden_states[, pos,]
            diff <- (r_pos - py_pos)$abs()$max()$item()
            cat(sprintf("Position %d diff: %.6f\n", pos, diff))
        }
    })

# ============================================================================
# Summary
# ============================================================================
cat("\n=== Summary ===\n")
cat(sprintf("Hidden states max diff:  %.6f\n", diff_hidden))
cat(sprintf("Hidden states mean diff: %.6f\n", mean_diff_hidden))
cat(sprintf("Speech logits max diff:  %.6f\n", diff_logits))
cat(sprintf("Speech logits mean diff: %.6f\n", mean_diff_logits))

# Check threshold
threshold <- 0.01# Looser threshold for Llama (many layers)
if (diff_hidden < threshold && diff_logits < threshold) {
    cat(sprintf("\nAll components match within threshold %.4f\n", threshold))
} else {
    cat(sprintf("\nSome components exceed threshold %.4f\n", threshold))
}

cat("\nDone.\n")

