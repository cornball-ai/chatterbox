## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission, so the one NOTE is the expected
  "New submission" from the CRAN incoming-feasibility check.

A spell-check may flag "TTS", "Resemble", and "HuggingFace"; these are a
correct acronym and proper names.

## Test environments

* local Ubuntu 24.04, R 4.6.0
* GitHub Actions (r-ci): ubuntu-latest, macos-latest
* win-builder: R-release and R-devel

## Notes

This package is a native R 'torch' port of Resemble AI's Chatterbox
text-to-speech model. Model weights (~2 GB) are downloaded from
HuggingFace on first use via the 'hfhub' package, only after explicit
user consent (`download_chatterbox_models()` prompts interactively, or
`options(chatterbox.consent = TRUE)`); nothing is downloaded at install,
load, example, test, or vignette time.

Examples that need the downloaded weights and a working 'torch'
installation (libtorch/Lantern) are wrapped in `\dontrun{}`, since neither
is available in the CRAN check environment. The remaining examples use
only base R and 'tuneR'. The test suite gates every 'torch'-dependent
test on `torch::torch_is_installed()`, so those tests skip cleanly where
libtorch is absent.
