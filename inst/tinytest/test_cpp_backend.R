# The C++ T3 decode routine must be registered. This guards the manual
# useDynLib(chatterbox, .registration = TRUE) line in NAMESPACE, which
# tinyrox::document() silently drops (it does not support @useDynLib).
# That exact failure shipped in PR #3 and went unnoticed for months:
# backend = "cpp" errored with '"cpp_t3_decode" not available'.
#
# Note: cpp_t3_decode is registered even in no-libtorch builds (the
# stub errors at call time), so this holds on machines without libtorch.

loadNamespace("chatterbox")
expect_true(is.loaded("cpp_t3_decode", PACKAGE = "chatterbox"))
