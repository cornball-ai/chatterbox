# Extract Python function/method signatures with defaults from the
# chatterbox-tts reference source using treesitR.
# Usage: r scripts/py_signatures.R /tmp/chatterbox-py > /tmp/py_signatures.txt

py_dir <- if (exists("argv") && length(argv) >= 1) argv[1] else "/tmp/chatterbox-py"

parser <- treesitR::ts_parser_new()
treesitR::ts_parser_set_language(parser, treesitR::ts_language_python())

walk_defs <- function (node, file, class_name = NULL)
{
    type <- treesitR::ts_node_type(node)
    if (type == "class_definition") {
        name_node <- treesitR::ts_node_child_by_field(node, "name")
        class_name <- treesitR::ts_node_text(name_node)
    }
    if (type == "function_definition") {
        name_node <- treesitR::ts_node_child_by_field(node, "name")
        params_node <- treesitR::ts_node_child_by_field(node, "parameters")
        fn <- treesitR::ts_node_text(name_node)
        params <- gsub("\\s+", " ", treesitR::ts_node_text(params_node))
        row <- treesitR::ts_node_start_point(node)[1]
        qual <- if (is.null(class_name)) fn else paste0(class_name, ".", fn)
        cat(sprintf("%s:%d %s%s\n", file, row + 1, qual, params))
    }
    n <- treesitR::ts_node_named_child_count(node)
    if (n > 0) {
        for (i in seq_len(n)) {
            walk_defs(treesitR::ts_node_named_child(node, i - 1), file, class_name)
        }
    }
}

files <- list.files(py_dir, pattern = "\\.py$", recursive = TRUE, full.names = TRUE)
for (f in files) {
    src <- paste(readLines(f, warn = FALSE), collapse = "\n")
    if (!nzchar(src)) next
    tree <- treesitR::ts_parse(parser, src)
    walk_defs(treesitR::ts_tree_root_node(tree), sub(paste0(py_dir, "/"), "", f))
}
