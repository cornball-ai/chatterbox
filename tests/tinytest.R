if (requireNamespace("tinytest", quietly = TRUE)) {
  # Redirect all three user-dir roots so neither this package nor any
  # dependency writes to the real home filespace during checks
  Sys.setenv(R_USER_CACHE_DIR  = tempfile("chatterbox_cache_"),
             R_USER_DATA_DIR   = tempfile("chatterbox_data_"),
             R_USER_CONFIG_DIR = tempfile("chatterbox_config_"))
  tinytest::test_package("chatterbox")
}
