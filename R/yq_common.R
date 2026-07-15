# Shared internal helpers for the anvl/yunque ports. One definition per
# name package-wide (two same-named helpers with different bodies once
# caused segfaults); verify scripts source this file first.

# [B, S, hidden] -> [B, n_head, S, head_dim]
.yq_heads <- function(x, batch, n_head, head_dim) {
  s <- anvl::shape(x)
  anvl::nv_transpose(anvl::nv_reshape(x, c(batch, s[2L], n_head, head_dim)),
    c(1L, 3L, 2L, 4L))
}

# Leaky ReLU (torch default negative slope 0.01); no yunque primitive.
.yq_leaky_relu <- function(x, slope = 0.01) {
  anvl::nv_max(x, 0) + anvl::nv_min(x, 0) * slope
}

# Round doubles to their nearest float32 values (kept as doubles). Used
# to mirror the reference's float32 scalar chains host-side (timestep
# schedules, sinusoid tables, source prep); single-rounding emulation
# via double is exact for +, -, *, /.
.yq_f32 <- function(x) {
  d <- dim(x)
  y <- readBin(writeBin(as.numeric(x), raw(), size = 4L), "numeric",
    size = 4L, n = length(x))
  if (!is.null(d)) {
    dim(y) <- d
  }
  y
}
