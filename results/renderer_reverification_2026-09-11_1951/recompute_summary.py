"""Recompute this renderer verification summary from preserved receipts, without a GPU.

Default: print JSON. --write: exclusively create summary.json; never overwrite a result.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent


def read(path):
    return json.loads(path.read_text())


def summarize():
    checks, stages = {}, {}
    for stage, seed in [('r3', 173), ('r4', 409), ('r4b', 409)]:
        paths = [HERE / (stage + '_run%d/run.json' % i) for i in (1, 2)]
        a, b = [read(p) for p in paths]
        previous = read(RESULTS / ('renderer_%s_seed%d_run1_2026-09-11/run.json' % (stage, seed)))
        fields = ['array_sha256', 'checks', 'run_verdict', 'camera', 'seed', 'generic_kernel_sha256']
        fields += ['metrics'] if stage == 'r3' else [
            'fixture', 'per_scene', 'summary', 'thresholds', 'camera_poses',
            'within_bin_ratio_lambertian_over_depth', 'within_bin_ratio_uniform_color_lambertian_over_depth']
        exact = {k: a[k] == previous[k] for k in fields}
        checks[stage + '_pair_byte_identical'] = paths[0].read_bytes() == paths[1].read_bytes()
        checks[stage + '_historical_outputs_exact'] = all(exact.values())
        checks[stage + '_runtime_unchanged'] = a['runtime'] == previous['runtime']
        stages[stage] = {'experimental_verdict': a['run_verdict'], 'compared_historical_fields': exact,
                         'current_commit': a['git_commit'], 'historical_commit': previous['git_commit'],
                         'metadata_note': 'Commit differs; current R4 additionally records fixture_name/stage.'}
    audits = {}
    for stage in ['r4', 'r4b']:
        value = read(HERE / (stage + '_numpy_audit.json'))
        checks[stage + '_numpy_audit'] = value['status'] == 'PASS' and all(value['checks'].values())
        audits[stage] = {'max_scalar_absolute_difference': max(value['absolute_differences'].values()),
                         'material_retention_per_scene': value['material_retention_per_scene']}
    rows = []
    expected_cells = {(w, h, b) for w, h in [(160,120), (240,135), (480,270)] for b in [1,8,32]}
    expected_cells |= {(480,270,64), (480,270,128)}
    all_sources = []
    for path in sorted((HERE / 'r5_direct').glob('*.json')):
        j = read(path)
        cell = (j['camera']['width'], j['camera']['height'], j['batch'])
        all_sources.append(j['provenance'])
        for arm in ['flat', 'lambertian']:
            samples = [s for r in j['records'] if r['mode'] == arm for s in r['samples_ms']]
            checks[path.stem + '_' + arm + '_count_mean'] = (
                len(samples) == 30 and abs(statistics.mean(samples) - j['arms'][arm]['mean_ms']) < 1e-10)
        telemetry = j['telemetry']
        checks[path.stem + '_telemetry'] = bool(telemetry['samples']) and not telemetry['errors']
        checks[path.stem + '_source_stable'] = j['source_unchanged_during_run']
        samples = telemetry['samples']
        device_mem = [float(g[3]) for s in samples for g in s['gpu_raw']]
        utilization = [float(g[2]) for s in samples for g in s['gpu_raw']]
        process_mem = [s['process_memory_mib'] for s in samples if s['process_memory_mib'] is not None]
        rows.append({'cell': list(cell), 'arms': j['arms'], 'ratio': j['lambertian_over_flat'],
                     'images_per_second_lambertian': j['images_per_second_lambertian'],
                     'torch_peak_allocated_mib': max(r['torch_peak_allocated_mib'] for r in j['records']),
                     'torch_peak_reserved_mib': max(r['torch_peak_reserved_mib'] for r in j['records']),
                     'sampled_device_memory_peak_mib': max(device_mem),
                     'sampled_process_memory_peak_mib': max(process_mem) if process_mem else None,
                     'sampled_device_utilization_percent_range': [min(utilization), max(utilization)],
                     'telemetry_sample_count': len(samples)})
    checks['all_11_direct_cells_present'] = {tuple(r['cell']) for r in rows} == expected_cells and len(rows) == 11
    checks['supplemental_source_manifest_identical'] = bool(all_sources) and all(v == all_sources[0] for v in all_sources)
    legacy = read(HERE / 'r5_legacy_large/benchmark.json')['rows'][-1]
    isolated = read(HERE / 'r5_legacy_isolated128/benchmark.json')['rows'][0]
    delta = legacy['torch_peak_allocated_mib'] - isolated['torch_peak_allocated_mib']
    # Owned GBuffer: range/depth/normal/face/instance/valid = 4+4+12+4+4+1 bytes/pixel.
    retained = 64 * 480 * 270 * 29 / 1024**2
    checks['memory_delta_equals_previous_gbuffer'] = abs(delta-retained) < 1e-10
    log = (HERE / 'full_tests_gpu_visible.log').read_text()
    count = int(re.search(r'Ran (\d+) tests', log).group(1))
    skips = int(re.search(r'OK \(skipped=(\d+)\)', log).group(1))
    checks['full_tests'] = count == 1415 and skips == 4 and '\nFAILED (' not in log
    preserved = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(HERE.rglob('*.json')) if p.name not in ('summary.json', 'commands.json')}
    return {'schema': 'renderer_reverification_summary_v1',
            'status': 'VERIFIED_WITH_LIMITATIONS' if all(checks.values()) else 'VERIFICATION_MISMATCH',
            'meaning': 'Verification status is not R4 acceptance; R4/R4b remain FAIL.',
            'checks': checks, 'stages': stages, 'numpy_audits': audits, 'direct_r5': rows,
            'legacy_r5_128_ratio': legacy['lambertian_over_flat'],
            'memory_audit': {'legacy_chained_mib': legacy['torch_peak_allocated_mib'],
                             'legacy_isolated_mib': isolated['torch_peak_allocated_mib'],
                             'delta_mib': delta, 'previous_64_scene_gbuffer_mib': retained},
            'tests': {'run': count, 'passed': count-skips, 'skipped': skips, 'failures': 0, 'errors': 0},
            'supplemental_provenance': all_sources[0],
            'runtime': read(HERE / 'r5_direct/480x270_b128.json')['runtime'],
            'receipt_sha256': preserved,
            'limitations': ['Fixed two-box fixture only; R4/R4b remain FAIL.',
                            'Direct timing includes periodic nvidia-smi telemetry and shared desktop load.',
                            'Short-cell means have outliers; ratios below one are not a shading speedup claim.',
                            'Sampled VRAM maxima are not instantaneous peaks; Torch memory is not total VRAM.',
                            'No universal slowdown bound, causal shortcut claim, or simulator step rate.',
                            'Supplemental tool/protocol were uncommitted and content-hash pinned.']}


if __name__ == '__main__':
    result = summarize()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if '--write' in sys.argv:
        with (HERE / 'summary.json').open('x') as stream:
            stream.write(text)
    print(text)
    if result['status'] != 'VERIFIED_WITH_LIMITATIONS':
        raise SystemExit(1)
