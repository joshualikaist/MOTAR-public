# ETH current-tree release validation

Code-complete source: `d6bccb49931e61d3f0b26c1f58ffad96588ba418`.
Only release documentation and captured validation evidence follow that commit.
No scientific results were recalculated. No policy evaluation/training was launched.

[Receipt](validation_receipt.json) binds the logs by SHA-256 and lists commands and test counts.
[Current-tree audit](current_tree_audit.json) records source commit, actual source-file hash,
working-tree dirtiness (new validation output was untracked), archive members and byte preservation.
Historical reachability remains OPEN; no history rewrite took place.

## Commands used

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B -m unittest discover -s tests -v
/tmp/motar-release-validation-6rTnrm/venv/bin/python -B tools/check_public_docs.py --schema
/tmp/motar-release-validation-6rTnrm/venv/bin/cffconvert --validate
node tests/test_status_site.js
node tests/test_public_status_manifest.js
node tests/test_status_arena_route.js
node tests/test_status_arena_motion.js
node tests/test_status_webgl_headless.js
python tools/audit_external_data_release.py audit --output /path/to/new-audit.json
```

The two absolute interpreter paths identify the local measured environments; on another host use
the repository's documented simulator test and separate public-validation environments, not literal
copies of these host paths. The audit output must not already exist. It is read-only apart from
writing that new report. The full suite is the repository's canonical unittest suite, **not pytest**;
pytest was not installed in the historical environment and was not added by this task.

The first unit log reports 1,882 tests in 66.638 s, with 4 pre-existing skips and no failures/errors.
The existing suite's GPU-dependent unit checks used the existing device; no GPU evaluation campaign
was started. Subsequent editorial-only changes are covered by the final public-doc/schema checks.

The removed-link audit scans Markdown inline links and HTML/SVG src/href after excluding fenced
code, resolving local targets against all 14 exact removed paths. Historical plain path/hash strings
are deliberately retained. Its scope is removal-induced dead references, not all historic links.

Scientific preservation: the inventory pins 46 existing non-image ETH files. All 45 immutable
entries match before/after SHA. Only the explicitly declared handoff note changed its availability
text/two live links. All experiment JSON, CSV, summaries and original source files remain identical.
