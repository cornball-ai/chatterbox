# Safetensors dtype string -> Rtorch dtype code mapping
.safe_dtype_map <- c(
  "F16" = 5L,   # torch_float16
  "BF16" = 15L, # torch_bfloat16
  "F32" = 6L,   # torch_float32
  "F64" = 7L,   # torch_float64
  "I8" = 1L,    # torch_int8
  "I16" = 2L,   # torch_int16
  "I32" = 3L,   # torch_int32
  "I64" = 4L,   # torch_int64
  "U8" = 0L,    # torch_uint8
  "BOOL" = 11L  # torch_bool
)

#' Read safetensors file
#'
#' Loads tensors from a safetensors file as Rtorch tensors.
#' @param path Path to .safetensors file
#' @param device Device to load tensors to (ignored, CPU only)
#' @return Named list of torch tensors
#' @export
read_safetensors <- function(path, device = "cpu") {
  # Open the safetensors file and parse header
  con <- file(path, "rb")
  on.exit(close(con))

  # Read header length (8-byte little-endian uint64)
  header_len_raw <- readBin(con, "integer", n = 2L, size = 4L,
                            endian = "little")
  header_len <- header_len_raw[1]  # Assuming < 2^31

  # Read and parse JSON header
  header_raw <- readBin(con, "raw", n = header_len)
  header_json <- rawToChar(header_raw)
  header <- jsonlite::fromJSON(header_json, simplifyVector = FALSE)

  # Data starts right after 8-byte length + header
  data_start <- 8L + header_len

  # Remove __metadata__ if present
  header[["__metadata__"]] <- NULL

  # Build output list
  output <- vector("list", length(header))
  names(output) <- names(header)

  for (name in names(header)) {
    meta <- header[[name]]
    dtype_str <- meta$dtype
    shape <- as.integer(unlist(meta$shape))
    offsets <- as.integer(unlist(meta$data_offsets))
    offset_start <- data_start + offsets[1]
    offset_length <- offsets[2] - offsets[1]

    # Read raw bytes for this tensor
    seek(con, offset_start)
    raw_bytes <- readBin(con, "raw", n = offset_length)

    # Map dtype
    dtype_code <- .safe_dtype_map[dtype_str]
    if (is.na(dtype_code)) {
      stop(sprintf("Unsupported safetensors dtype: %s", dtype_str))
    }

    # Handle scalar tensors (empty shape)
    if (length(shape) == 0L) shape <- integer(0)

    # Create Rtorch tensor from raw bytes
    output[[name]] <- Rtorch::torch_tensor_from_buffer(raw_bytes, shape, dtype_code)
  }

  output
}
