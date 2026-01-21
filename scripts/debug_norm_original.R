#!/usr/bin/env Rscript
# Check original tfmr.norm.weight

rhydrogen::load_all()

cat("=== Loading T3 Weights ===\n")
paths <- download_chatterbox_models()
t3_weights <- read_safetensors(paths$t3_cfg, device = "cpu")

# Check original key
cat("\n=== Checking tfmr.norm.weight in original file ===\n")
if ("tfmr.norm.weight" %in% names(t3_weights)) {
  original <- t3_weights[["tfmr.norm.weight"]]
  cat("Found tfmr.norm.weight\n")
  cat("Shape:", paste(dim(original), collapse = "x"), "\n")
  cat("Mean:", original$mean()$item(), "\n")
  cat("Range: [", original$min()$item(), ", ", original$max()$item(), "]\n", sep = "")
  cat("First 10 values:", paste(round(as.numeric(original[1:10]$cpu()), 4), collapse = ", "), "\n")
} else {
  cat("tfmr.norm.weight NOT found\n")
}

# Also check if there's a model.norm.weight
cat("\n=== Other possible norm keys ===\n")
for (key in names(t3_weights)) {
  if (grepl("norm\\.weight$", key) && !grepl("layer", key)) {
    cat(key, ": mean =", t3_weights[[key]]$mean()$item(), "\n")
  }
}
