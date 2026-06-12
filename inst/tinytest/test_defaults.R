# chatterbox_defaults tier logic (no GPU or weights needed)

d6 <- chatterbox::chatterbox_defaults(vram_gb = 6)
expect_inherits(d6, "chatterbox_defaults")
expect_identical(d6$device, "cuda")
expect_equal(d6$options$torch.cuda_allocator_reserved_rate, 0.75)
expect_identical(d6$backend, "jit")
expect_true(d6$measured)

d8 <- chatterbox::chatterbox_defaults(vram_gb = 8)
expect_equal(d8$options$torch.cuda_allocator_reserved_rate, 0.6)
expect_false(d8$measured)

d12 <- chatterbox::chatterbox_defaults(vram_gb = 12)
expect_equal(d12$options$torch.cuda_allocator_reserved_rate, 0.5)
expect_false(d12$measured)

d16 <- chatterbox::chatterbox_defaults(vram_gb = 16)
expect_equal(d16$options$torch.cuda_allocator_reserved_rate, 0.5)
expect_true(d16$measured)

dcpu <- chatterbox::chatterbox_defaults(vram_gb = 0)
expect_identical(dcpu$device, "cpu")
expect_identical(dcpu$backend, "r")
expect_identical(dcpu$options, list())

# cards under 5 GB cannot hold the ~4.6 GB model: treated as CPU
expect_identical(chatterbox::chatterbox_defaults(vram_gb = 2)$device, "cpu")
expect_identical(chatterbox::chatterbox_defaults(vram_gb = 4)$device, "cpu")
expect_identical(chatterbox::chatterbox_defaults(vram_gb = 4.9)$device, "cpu")

# 5-5.5 GB runs CUDA but is a projection, not the measured 6 GB tier
d5 <- chatterbox::chatterbox_defaults(vram_gb = 5)
expect_identical(d5$device, "cuda")
expect_false(d5$measured)

# 13 GB sits between measured tiers: projected
expect_false(chatterbox::chatterbox_defaults(vram_gb = 13)$measured)

# print methods run and return invisibly
expect_stdout(print(d6), "jit")
expect_stdout(print(d6), "0.75")
expect_stdout(print(d8), "projected")
expect_stdout(print(dcpu), "CPU-only")

# the GC option is applicable directly
expect_silent(do.call(options, d16$options))
