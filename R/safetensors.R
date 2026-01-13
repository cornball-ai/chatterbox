# Safetensors file reader for chatteRbox
# Parses the safetensors format and loads tensors into torch

#' Read safetensors file
#'
#' @param path Path to .safetensors file
#' @param device Device to load tensors to ("cpu", "cuda", etc.)
#' @return Named list of torch tensors
#' @export
read_safetensors <- function(path, device = "cpu") {
  con <- file(path, "rb")
  on.exit(close(con))

  # Read header size (8 bytes, little-endian uint64)
  header_size_bytes <- readBin(con, "raw", n = 8)
  header_size <- sum(as.numeric(header_size_bytes) * (256^(0:7)))

  # Read and parse JSON header
  header_json <- readChar(con, header_size, useBytes = TRUE)
  header <- jsonlite::fromJSON(header_json, simplifyVector = FALSE)

  # Remove metadata entry if present
  header[["__metadata__"]] <- NULL

  # Filter out entries with empty names or NULL values
  # (jsonlite parses empty string keys but returns NULL values)
  valid_names <- names(header)[nchar(names(header)) > 0 & !sapply(header, is.null)]
  header <- header[valid_names]

  # Read the entire data buffer
  data_buffer <- readBin(con, "raw", n = file.info(path)$size - 8 - header_size)

  # Parse each tensor
  tensors <- list()

  for (name in names(header)) {
    tensor_info <- header[[name]]
    dtype <- tensor_info$dtype
    shape <- as.integer(unlist(tensor_info$shape))
    offsets <- as.numeric(unlist(tensor_info$data_offsets))

    start_byte <- offsets[1] + 1  # R is 1-indexed
    end_byte <- offsets[2]

    tensor_bytes <- data_buffer[start_byte:end_byte]
    tensors[[name]] <- bytes_to_tensor(tensor_bytes, dtype, shape, device)
  }

  tensors
}

#' Convert raw bytes to torch tensor
#'
#' @param bytes Raw vector of bytes
#' @param dtype Safetensors dtype string
#' @param shape Integer vector of tensor dimensions
#' @param device Target device
#' @return torch tensor
bytes_to_tensor <- function(bytes, dtype, shape, device) {
  # Determine R type and torch dtype
  type_info <- get_dtype_info(dtype)

  # Convert bytes to numeric vector
  n_elements <- prod(shape)
  if (length(shape) == 0) n_elements <- 1

  values <- readBin(bytes, what = type_info$r_type, n = n_elements,
                    size = type_info$size, endian = "little")

  # Handle bfloat16 specially
  if (dtype == "BF16") {
    values <- bfloat16_to_float32(bytes)
  }

  # Handle float16 specially
  if (dtype == "F16") {
    values <- float16_to_float32(bytes)
  }

  # Create tensor with appropriate shape
  if (length(shape) == 0) {
    tensor <- torch::torch_tensor(values, dtype = type_info$torch_dtype)
  } else {
    tensor <- torch::torch_tensor(values, dtype = type_info$torch_dtype)
    tensor <- tensor$view(shape)
  }

  tensor$to(device = device)
}

#' Get dtype information
#'
#' @param dtype Safetensors dtype string
#' @return List with r_type, size, torch_dtype
get_dtype_info <- function(dtype) {
  switch(dtype,
    "F32" = list(r_type = "double", size = 4, torch_dtype = torch::torch_float32()),
    "F64" = list(r_type = "double", size = 8, torch_dtype = torch::torch_float64()),
    "F16" = list(r_type = "raw", size = 2, torch_dtype = torch::torch_float32()),  # Convert to f32
    "BF16" = list(r_type = "raw", size = 2, torch_dtype = torch::torch_float32()), # Convert to f32
    "I8" = list(r_type = "integer", size = 1, torch_dtype = torch::torch_int8()),
    "I16" = list(r_type = "integer", size = 2, torch_dtype = torch::torch_int16()),
    "I32" = list(r_type = "integer", size = 4, torch_dtype = torch::torch_int32()),
    "I64" = list(r_type = "double", size = 8, torch_dtype = torch::torch_int64()),
    "U8" = list(r_type = "integer", size = 1, torch_dtype = torch::torch_uint8()),
    "BOOL" = list(r_type = "logical", size = 1, torch_dtype = torch::torch_bool()),
    stop("Unsupported dtype: ", dtype)
  )
}

#' Convert bfloat16 bytes to float32 values
#'
#' @param bytes Raw vector of bfloat16 bytes
#' @return Numeric vector of float32 values
bfloat16_to_float32 <- function(bytes) {
  n_elements <- length(bytes) %/% 2
  values <- numeric(n_elements)

  for (i in seq_len(n_elements)) {
    # Read 2 bytes as bfloat16
    bf16_bytes <- bytes[((i - 1) * 2 + 1):(i * 2)]

    # bfloat16 is the upper 16 bits of float32
    # So we just pad with zeros on the right
    f32_bytes <- c(raw(2), bf16_bytes)

    values[i] <- readBin(f32_bytes, "double", n = 1, size = 4, endian = "little")
  }

  values
}

#' Convert float16 bytes to float32 values
#'
#' @param bytes Raw vector of float16 bytes
#' @return Numeric vector of float32 values
float16_to_float32 <- function(bytes) {
  n_elements <- length(bytes) %/% 2
  values <- numeric(n_elements)

  for (i in seq_len(n_elements)) {
    # Read 2 bytes as uint16
    idx <- (i - 1) * 2
    h <- as.integer(bytes[idx + 1]) + as.integer(bytes[idx + 2]) * 256L

    # Extract float16 components
    sign <- bitwAnd(bitwShiftR(h, 15L), 1L)
    exponent <- bitwAnd(bitwShiftR(h, 10L), 0x1FL)
    mantissa <- bitwAnd(h, 0x3FFL)

    if (exponent == 0) {
      # Subnormal or zero
      if (mantissa == 0) {
        values[i] <- if (sign == 1) -0.0 else 0.0
      } else {
        # Subnormal: denormalized number
        values[i] <- ((-1)^sign) * (mantissa / 1024) * (2^(-14))
      }
    } else if (exponent == 31) {
      # Inf or NaN
      if (mantissa == 0) {
        values[i] <- if (sign == 1) -Inf else Inf
      } else {
        values[i] <- NaN
      }
    } else {
      # Normalized number
      values[i] <- ((-1)^sign) * (1 + mantissa / 1024) * (2^(exponent - 15))
    }
  }

  values
}

#' Load PyTorch .pt file (for conds.pt)
#'
#' This is a simplified loader for the specific structure of conds.pt
#' Full pickle parsing is complex; this handles the common case
#'
#' @param path Path to .pt file
#' @param device Device to load to
#' @return Loaded object (typically a list/dict structure)
#' @export
read_torch_pt <- function(path, device = "cpu") {
  # For now, use torch's built-in loader if available
  # This requires torch to be built with the loader
  tryCatch({
    torch::torch_load(path, device = device)
  }, error = function(e) {
    warning("Could not load .pt file directly. ",
            "Consider converting to safetensors format.")
    NULL
  })
}
