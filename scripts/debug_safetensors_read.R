#!/usr/bin/env Rscript
# Debug safetensors reading for a specific tensor

rhydrogen::load_all()

path <- "/home/troy/.cache/chatterbox/ResembleAI--chatterbox/t3_cfg.safetensors"

cat("=== Reading safetensors file ===\n")
con <- file(path, "rb")

# Read header size
header_size_bytes <- readBin(con, "raw", n = 8)
header_size <- sum(as.numeric(header_size_bytes) * (256^(0:7)))
cat("Header size:", header_size, "bytes\n")

# Read header
header_json <- readChar(con, header_size, useBytes = TRUE)
header <- jsonlite::fromJSON(header_json, simplifyVector = FALSE)

# Check tfmr.norm.weight info
cat("\n=== tfmr.norm.weight info ===\n")
norm_info <- header[["tfmr.norm.weight"]]
if (!is.null(norm_info)) {
  cat("dtype:", norm_info$dtype, "\n")
  cat("shape:", paste(unlist(norm_info$shape), collapse = "x"), "\n")
  cat("data_offsets:", paste(unlist(norm_info$data_offsets), collapse = ", "), "\n")

  # Read the data
  data_start <- 8 + header_size
  offsets <- as.numeric(unlist(norm_info$data_offsets))

  seek(con, data_start + offsets[1])
  tensor_bytes <- readBin(con, "raw", n = offsets[2] - offsets[1])

  cat("\nRaw bytes (first 40):", paste(head(as.integer(tensor_bytes), 40), collapse = ", "), "\n")
  cat("Total bytes:", length(tensor_bytes), "\n")

  # Try to read as float32
  n_elements <- 1024
  values <- readBin(tensor_bytes, what = "double", n = n_elements, size = 4, endian = "little")
  cat("\nValues as F32 (first 10):", paste(round(values[1:10], 4), collapse = ", "), "\n")
  cat("Values mean:", mean(values), "\n")
} else {
  cat("tfmr.norm.weight NOT FOUND in header\n")
}

close(con)

# Also check with alternative approach - read whole file
cat("\n=== Using read_safetensors function ===\n")
tensors <- read_safetensors(path, device = "cpu")

if ("tfmr.norm.weight" %in% names(tensors)) {
  norm_tensor <- tensors[["tfmr.norm.weight"]]
  cat("Tensor shape:", paste(dim(norm_tensor), collapse = "x"), "\n")
  cat("Tensor mean:", norm_tensor$mean()$item(), "\n")
  cat("Tensor first 10:", paste(round(as.numeric(norm_tensor[1:10]$cpu()), 4), collapse = ", "), "\n")
} else {
  cat("tfmr.norm.weight NOT in read tensors\n")
}
