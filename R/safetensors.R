#' Read safetensors file
#'
#' @param path Path to .safetensors file
#' @param device Device to load tensors to ("cpu", "cuda", etc.)
#' @return Named list of torch tensors
#' @keywords internal
read_safetensors <- function(path, device = "cpu") {
    safetensors::safe_load_file(path, framework = "torch", device = device)
}
