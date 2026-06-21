# .serve_resolve_voice: library names must win over like-named entries in
# the working directory, and only regular files count as path references.

resolve <- chatterbox:::.serve_resolve_voice

vd <- tempfile("voices_")
dir.create(vd)
barry <- file.path(vd, "Barry.mp3")
writeLines("not really audio", barry)        # content irrelevant to resolution
casey <- file.path(vd, "ShortCasey.wav")
writeLines("x", casey)

# Bare library name resolves to the library file.
expect_equal(resolve("Barry", vd), barry)
expect_equal(resolve("shortcasey", vd), casey) # case-insensitive

# A directory named like the voice in the working dir must NOT shadow it.
cwd <- tempfile("cwd_")
dir.create(cwd)
old <- setwd(cwd)
on.exit(setwd(old), add = TRUE)
dir.create("Barry")                            # the bug: file.exists("Barry") was TRUE
expect_equal(resolve("Barry", vd), barry)      # still the library file, not "Barry"/
setwd(old)

# An explicit path to a regular file is returned as-is.
expect_equal(resolve(barry, vd), barry)

# A directory path is not a valid reference.
expect_null(resolve(vd, vd))

# Unknown name resolves to NULL.
expect_null(resolve("Nonexistent", vd))

unlink(vd, recursive = TRUE)
