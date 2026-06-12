# split_text_chunks: sentence splitting with chunk_size enforcement
# (chunk_size was previously dead code; run-on sentences passed whole)

sc <- chatterbox:::split_text_chunks

# Multi-sentence text splits at sentence boundaries
out <- sc("One sentence. Another one. A third!", 200L)
expect_equal(length(out), 3L)
expect_equal(out[1], "One sentence.")

# Short single sentence stays whole
expect_equal(sc("Just one short sentence.", 200L), "Just one short sentence.")

# Run-on single sentence longer than chunk_size splits at commas
runon <- paste0("Yes, Rarely or never Almost never, at most once in a while, ",
    "over the past week Sometimes Only a couple of days over the past week, ",
    "not many times in any given day Often Four or more days over the past week.")
out <- sc(runon, 100L)
expect_true(length(out) >= 2L)
expect_true(all(nchar(out) <= 110L | !grepl(",", out)))
# Nothing lost: rejoining covers the original words
expect_equal(gsub("[ ]+", " ", paste(out, collapse = " ")),
             gsub("[ ]+", " ", runon))

# A comma-less giant clause stays whole (no mid-clause splitting)
giant <- paste(rep("word", 80), collapse = " ")
expect_equal(sc(giant, 100L), giant)

# Empty-ish input produces no chunks
expect_equal(length(sc("   ", 200L)), 0L)
