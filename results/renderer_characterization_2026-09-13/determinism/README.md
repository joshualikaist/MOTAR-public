# RC determinism gate

Verdict: **DETERMINISTIC**

The exporter ran twice in two independent OS processes with identical arguments. Every
array sha256 is compared, and the metadata is compared apart from the runtime fingerprint
and the output path.

Arrays compared: `depth_m`, `face_id`, `instance_id`, `normal_world`, `range_m`, `rgb`, `valid`

Differing arrays: none

