#!/usr/bin/env Rscript
# Debug weight keys

rhydrogen::load_all()

cat("=== Loading T3 Weights ===\n")
paths <- download_chatterbox_models()
t3_weights <- read_safetensors(paths$t3_cfg, device = "cpu")

# Extract Llama weights with same logic as load_t3_weights
cat("\n=== Extracting Llama Weights ===\n")
llama_weights <- list()
for (name in names(t3_weights)) {
    if (startsWith(name, "tfmr.")) {
        new_name <- sub("^tfmr\\.", "", name)
        llama_weights[[new_name]] <- t3_weights[[name]]
    }
}

cat("Total llama weight keys:", length(llama_weights), "\n")

# Check for norm weight
cat("\n=== Checking norm weight ===\n")
if ("norm.weight" %in% names(llama_weights)) {
    cat("Found norm.weight in llama_weights\n")
    cat("Shape:", paste(dim(llama_weights[["norm.weight"]]), collapse = "x"), "\n")
    cat("Mean:", llama_weights[["norm.weight"]]$mean()$item(), "\n")
} else {
    cat("norm.weight NOT found in llama_weights\n")
    cat("Looking for similar keys:\n")
    for (key in names(llama_weights)) {
        if (grepl("norm", key)) {
            cat("  ", key, "\n")
        }
    }
}

# Check the exact key parsing
cat("\n=== Testing key parsing ===\n")
test_key <- "norm.weight"
parts <- strsplit(test_key, "\\.") [[1]]
cat("Parts:", paste(parts, collapse = ", "), "\n")
cat("parts[1] == 'norm':", parts[1] == "norm", "\n")

# Check for all keys with 'norm'
cat("\n=== All keys containing 'norm' in llama_weights ===\n")
norm_keys <- grep("norm", names(llama_weights), value = TRUE)
for (key in norm_keys) {
    cat("  ", key, "\n")
}

