# VOID — controlled-heading guard false positive

The first `toward` cell stopped during task construction before producing any episode or result
JSON. The evaluator uses `NAVRL_GENERAL_TRAIN=1` to reproduce checkpoint provenance while the
actual execution authority is bulk evaluation. The initial guard inspected the provenance flag
alone and therefore rejected this valid diagnostic as training.

This directory is retained only as failure evidence. It is excluded from every quantitative
result. Commit `f55bf28` corrected the guard to require bulk-evaluation mode, a non-empty bulk
result path, and an evaluation checkpoint. The valid four-cell run is in the sibling directory
`navrl_ref5in_cv_heading_seed337/`.
