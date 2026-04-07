library(chatterbox)

# chatterbox() returns an object of class "chatterbox" with expected slots
m <- chatterbox(device = "cpu")
expect_inherits(m, "chatterbox")
expect_equal(m$device, "cpu")
expect_false(m$turbo)
expect_false(m$loaded)
expect_null(m$t3)
expect_null(m$s3gen)

# turbo flag is plumbed through
m_turbo <- chatterbox(device = "cpu", turbo = TRUE)
expect_true(m_turbo$turbo)
