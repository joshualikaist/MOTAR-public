# D8-A attempt 1 — `VOID_EXECUTION`

This directory is preserved because the preregistered runner opened it on clean implementation
commit `8fdeed94bf419b3cfaeb97d05efdf17067db65dd`.

No pose, renderer, or cost cell executed.  Isaac Gym stopped during import because the selected
conda interpreter's `ninja` executable was not on the inherited `PATH`.  The Python package was
installed, but `torch.utils.cpp_extension` checks for the executable.  Therefore this attempt is
an environment-launch failure, not a D8 geometry/correctness/cost result, and its recorded
`TECHNICAL_NO_GO_UNFINISHED` must not be pooled with a completed D8-A run.

The failure receipt is [`failure.json`](failure.json).  A retry must use a new result directory,
must bind and record the `ninja` executable beside the selected interpreter, and must not modify
this directory.
