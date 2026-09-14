"""P8: frozen validation calibration, single-GT size conditions, observed-time transitions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from perception_candidates import read_jsonl, sha256_file
from perception_temporal import candidate_box, load_aligned_records
from train_perception_temporal import association_metrics
from runtime_fingerprint import runtime_fingerprint

ROOT = Path(__file__).resolve().parents[1]
STATES = ('HIT', 'FALSE_LOCK', 'NO_LOCK')
EDGES = (0, 8, 16, 32, 64)
QUANTILES = (0, .05, .25, .5, .75, .95, .99, 1)
MAX_GAP_S = .5


def size_bin(size):
    if not math.isfinite(size) or size <= 0:
        raise ValueError('GT extent must be finite and positive')
    return sum(size >= edge for edge in EDGES[1:])


def distribution(values):
    if not values:
        return {'n': 0, 'mean': None, 'quantiles': None}
    if not np.isfinite(values).all():
        raise ValueError('nonfinite distribution')
    return {'n': len(values), 'mean': float(np.mean(values)),
            'quantiles': {str(q): float(np.quantile(values, q)) for q in QUANTILES}}


def transition_summary(rows, states):
    counts = [[sum(a == left and b == right for a, b, _ in rows) for right in states]
              for left in states]
    return {'states': list(states), 'counts': counts,
            'probabilities': [[v / sum(row) for v in row] if sum(row) else None for row in counts],
            'row_supported': [sum(row) >= 10 for row in counts],
            'interval_seconds': distribution([dt for _, _, dt in rows]),
            'samples': [list(row) for row in rows]}


def segments(frames):
    result, current = [], []
    for row in frames:
        if not row['conditional_eligible']:
            if current: result.append(current)
            current = []
            continue
        if current:
            previous = current[-1]
            dt = (row['timestamp_ns'] - previous['timestamp_ns']) / 1e9
            if row['sequence'] != previous['sequence'] or dt > MAX_GAP_S:
                result.append(current)
                current = []
            elif dt <= 0:
                raise ValueError('nonmonotonic conditional timestamps')
        current.append(row)
    if current: result.append(current)
    return result


def bursts_for_segment(segment, loss=False):
    result, start = [], 0
    def state(row):
        return ('HIT' if row['state'] == 'HIT' else 'LOST') if loss else row['state']
    for end in range(1, len(segment) + 1):
        if end < len(segment) and state(segment[end]) == state(segment[start]):
            continue
        first = segment[start]
        right_censored = end == len(segment)
        last_time = segment[end - 1 if right_censored else end]['timestamp_ns']
        result.append({'kind': 'target_loss' if loss else 'state', 'state': state(first),
                       'sequence': first['sequence'], 'start_frame_id': first['frame_id'],
                       'size_bin_at_start': first['size_bin'], 'observations': end - start,
                       'duration_seconds': (last_time - first['timestamp_ns']) / 1e9,
                       'left_censored': start == 0, 'right_censored': right_censored,
                       'is_reacquisition': loss and state(first) == 'LOST' and start > 0 and not right_censored})
        start = end
    return result


def summarize(frames):
    parts = segments(frames)
    bursts = [burst for part in parts for loss in (False, True)
              for burst in bursts_for_segment(part, loss)]
    bins = []
    for index, lower in enumerate(EDGES):
        subset = [f for f in frames if f['size_bin'] == index]
        transitions = []
        for part in parts:
            for left, right in zip(part, part[1:]):
                if left['size_bin'] == index:
                    transitions.append((left['state'], right['state'],
                                        (right['timestamp_ns'] - left['timestamp_ns']) / 1e9))
        state_counts = dict(Counter(f['state'] for f in subset))
        errors = {}
        for state in STATES[:2]:
            samples = [f for f in subset if f['state'] == state]
            errors[state] = {
                'frame_ids': [f['frame_id'] for f in samples],
                'joint_du_dv_px_samples': [f['center_offset_px'] for f in samples],
                'joint_du_dv_normalized_samples': [f['center_offset_normalized'] for f in samples],
                'radial_error_px': distribution([math.hypot(*f['center_offset_px']) for f in samples])}
        by_state_bursts = {}
        for label in (*STATES, 'LOST'):
            relevant = [b for b in bursts if b['size_bin_at_start'] == index and b['state'] == label
                        and b['kind'] == ('target_loss' if label == 'LOST' else 'state')]
            completed = [b['duration_seconds'] for b in relevant
                         if not b['left_censored'] and not b['right_censored']]
            by_state_bursts[label] = {'observed_runs': len(relevant),
                'left_censored': sum(b['left_censored'] for b in relevant),
                'right_censored': sum(b['right_censored'] for b in relevant),
                'completed_duration_seconds': distribution(completed),
                'completed_duration_samples_seconds': completed}
        selected_counts = state_counts.get('HIT', 0) + state_counts.get('FALSE_LOCK', 0)
        bins.append({'index': index, 'lower_inclusive_px': lower,
            'upper_exclusive_px': EDGES[index + 1] if index + 1 < len(EDGES) else None,
            'frames': len(subset), 'supported': len(subset) >= 30,
            'sequence_counts': dict(Counter(f['sequence'] for f in subset)),
            'resolution_counts': dict(Counter(f"{f['width_px']}x{f['height_px']}" for f in subset)),
            'state_counts': {s: state_counts.get(s, 0) for s in STATES},
            'state_probabilities': {s: state_counts.get(s, 0) / len(subset) if subset else None for s in STATES},
            'false_lock_rate_among_selected': state_counts.get('FALSE_LOCK', 0) / selected_counts if selected_counts else None,
            'error': errors, 'transition_3state': transition_summary(transitions, STATES),
            'transition_lock_no_lock': transition_summary([
                ('NO_LOCK' if a == 'NO_LOCK' else 'LOCK', 'NO_LOCK' if b == 'NO_LOCK' else 'LOCK', dt)
                for a, b, dt in transitions], ('LOCK', 'NO_LOCK')),
            'bursts': by_state_bursts,
            'latency_with_decode_ms': distribution([v for f in subset for v in f['latency_ms_repeats']]),
            'latency_by_repeat_ms': [distribution([f['latency_ms_repeats'][i] for f in subset]) for i in range(3)]})
    return {'schema_version': 'motar.perception-error-model.v1',
        'calibration_split': 'validation', 'conditional_population': 'exactly_one_GT',
        'size_definition': 'sqrt(original_pixel_GT_area)', 'test_used': False,
        'states': list(STATES), 'max_segment_gap_seconds': MAX_GAP_S,
        'excluded_gt_count_histogram': dict(Counter(str(f['gt_count']) for f in frames if not f['conditional_eligible'])),
        'frames': len(frames), 'eligible_frames': sum(f['conditional_eligible'] for f in frames),
        'segments': len(parts), 'transition_time_semantics': 'one empirical observation interval; not simulator tick',
        'reacquisition_seconds': distribution([b['duration_seconds'] for b in bursts if b['is_reacquisition']]),
        'bins': bins}, bursts


def checked_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected: raise ValueError('hash mismatch: ' + str(path))
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--s1-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing existing output')
    d = args.data_root
    manifest = d / 'nps_val_sequence.jsonl'
    candidates = d / 'nps_val_candidates.jsonl.gz'
    aligned, meta, cmeta = load_aligned_records(manifest, str(manifest) + '.receipt.json',
                                               candidates, str(candidates) + '.receipt.json')
    if meta['split'] != 'val': raise ValueError('validation only')
    receipt_path = args.run / 'receipt.json'
    run = json.loads(receipt_path.read_text())
    checked_hash(manifest, run['validation_manifest_sha256'])
    checked_hash(candidates, run['validation_candidates_sha256'])
    checked_hash(args.run / 'best.pt', run['checkpoint_sha256'])
    pred_path = args.run / 'validation_predictions.jsonl.gz'
    checked_hash(pred_path, run['validation_predictions_sha256'])
    if run['detector_weights_sha256'] != cmeta['weights_sha256']:
        raise ValueError('detector provenance mismatch')
    predictions = list(read_jsonl(pred_path))
    if len(predictions) != len(aligned): raise ValueError('prediction count mismatch')
    s1_expected = json.loads((ROOT / 'results/perception_streaming_overlap_s1_2026-09-09/summary.json').read_text())
    inputs = {str(p): sha256_file(p) for p in (manifest, candidates, receipt_path, pred_path,
              Path(str(manifest) + '.receipt.json'), Path(str(candidates) + '.receipt.json'))}
    gpu_reports, timings = [], []
    for i in range(1, 4):
        name = f'overlap_{i}.json'
        path = args.s1_root / name
        inputs[str(path)] = checked_hash(path, s1_expected['raw_artifact_sha256'][name])
        report = json.loads(path.read_text())
        if (report['test_used'] or not report['overlap_flow'] or not report['metrics_exact_match']
                or report['rank_mismatches'] or report['candidate_frame_mismatches']
                or report['motion_frame_bit_mismatches']
                or report['checkpoint_sha256'] != run['checkpoint_sha256']
                or report['manifest_sha256'] != meta['manifest_sha256']):
            raise ValueError('invalid S1 input')
        tpath = Path(str(path) + '.timings.jsonl.gz')
        inputs[str(tpath)] = checked_hash(tpath, report['timings_sha256'])
        timing = list(read_jsonl(tpath))
        if len(timing) != len(aligned): raise ValueError('timing count mismatch')
        timings.append(timing)
        gpu_reports.append(report)
    ranks = []
    for row, prediction in zip(aligned, predictions):
        for key in ('frame_id', 'source_sequence_id', 'capture_timestamp_ns', 'frame_index'):
            if row['source'][key] != prediction[key]: raise ValueError('prediction identity mismatch')
        rank = prediction['predicted_rank']
        if rank is not None and (type(rank) is not int or not 0 <= rank < len(row['candidate_record']['candidates'])):
            raise ValueError('invalid predicted rank')
        ranks.append(5 if rank is None else rank)
    metrics, detailed = association_metrics(type('Rows', (), {'records': aligned})(), ranks, .3)
    if any(r['metrics'] != metrics for r in gpu_reports): raise ValueError('baseline metrics differ')
    frames = []
    for index, row in enumerate(aligned):
        s = row['source']; gt = s['ground_truth_xyxy']
        for box in gt:
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('invalid GT geometry')
        eligible = len(gt) == 1
        _, selected, box, outcome = detailed[index]
        state = 'HIT' if outcome['hit'] else ('FALSE_LOCK' if outcome['selected'] else 'NO_LOCK')
        offsets = [(box[0] + box[2] - gt[0][0] - gt[0][2]) / 2,
                   (box[1] + box[3] - gt[0][1] - gt[0][3]) / 2] if eligible and selected is not None else None
        latencies = []
        for timing in timings:
            if timing[index]['frame_id'] != s['frame_id']: raise ValueError('timing identity mismatch')
            value = timing[index]['latency_ms']['with_decode']
            if not math.isfinite(value) or value <= 0: raise ValueError('invalid latency')
            latencies.append(value)
        size = math.sqrt((gt[0][2]-gt[0][0])*(gt[0][3]-gt[0][1])) if eligible else None
        frames.append({'frame_id': s['frame_id'], 'sequence': s['source_sequence_id'],
            'timestamp_ns': s['capture_timestamp_ns'], 'width_px': s['width_px'], 'height_px': s['height_px'],
            'gt_count': len(gt), 'conditional_eligible': eligible, 'size_px': size,
            'size_bin': size_bin(size) if eligible else None, 'state': state,
            'center_offset_px': offsets,
            'center_offset_normalized': [offsets[0]/s['width_px'],offsets[1]/s['height_px']] if offsets else None,
            'latency_ms_repeats': latencies})
    model, bursts = summarize(frames)
    model['overall_any_gt_metrics'] = metrics
    args.output.mkdir(parents=True, exist_ok=False)
    def dump(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    dump('error_model.json', model)
    dump('bursts.json', bursts)
    (args.output / 'frames.jsonl').write_text(''.join(json.dumps(f, sort_keys=True, allow_nan=False)+'\n' for f in frames))
    dump('receipt.json', {'schema_version': 1, 'test_used': False, 'runtime': runtime_fingerprint(),
        'measurement_runtimes': [r['runtime'] for r in gpu_reports],
        'source_git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'source_sha256': sha256_file(__file__),
        'contract_sha256': sha256_file(ROOT / 'docs/plans/perception_p8_execution_2026-09-09.md'),
        'checkpoint_sha256': run['checkpoint_sha256'], 'input_sha256': inputs,
        'output_sha256': {name: sha256_file(args.output/name) for name in ('error_model.json','frames.jsonl','bursts.json')}})
    print(json.dumps({'frames': len(frames), 'eligible_frames': model['eligible_frames'],
                      'excluded': model['excluded_gt_count_histogram'],
                      'bin_counts': [b['frames'] for b in model['bins']]}))


if __name__ == '__main__': main()
