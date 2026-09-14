"""Validation-only candidate ceiling and selector error decomposition."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from perception_candidates import read_jsonl, sha256_file
from perception_temporal import box_iou, candidate_box, load_aligned_records


def audit(aligned, predictions, threshold=0.3):
    counts = Counter()
    clips = defaultdict(Counter)
    if len(aligned) != len(predictions):
        raise ValueError('prediction count mismatch')
    for item, prediction in zip(aligned, predictions):
        source = item['source']
        if prediction['frame_id'] != source['frame_id']:
            raise ValueError('prediction identity mismatch')
        overlaps = [max([box_iou(candidate_box(c), gt)
                        for gt in source['ground_truth_xyxy']] or [0.0])
                    for c in item['candidate_record']['candidates']]
        reachable = any(x >= threshold for x in overlaps)
        rank = prediction['predicted_rank']
        hit = rank is not None and 0 <= rank < len(overlaps) and overlaps[rank] >= threshold
        events = ['frames']
        if reachable:
            events.append('oracle_reachable')
            events.append('selected_hit' if hit else
                          'reachable_abstention' if rank is None else 'reachable_wrong_candidate')
        else:
            events.append('detector_miss')
            events.append('unreachable_abstention' if rank is None else 'unreachable_false_lock')
        if overlaps and overlaps[0] >= threshold:
            events.append('top1_hit')
        counts.update(events)
        clips[source['source_sequence_id']].update(events)
    return {'counts': dict(counts), 'per_clip': {k: dict(v) for k, v in clips.items()},
            'oracle_hit_rate': counts['oracle_reachable'] / counts['frames'],
            'oracle_utility_upper_bound': counts['oracle_reachable'] / counts['frames']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing existing output')
    root = args.data_root
    manifest = root / 'nps_val_sequence.jsonl'
    candidates = root / 'nps_val_candidates.jsonl.gz'
    meta = json.loads(Path(str(manifest) + '.receipt.json').read_text())
    if meta['split'] != 'val':
        raise ValueError('validation only')
    aligned, _, _ = load_aligned_records(manifest, str(manifest)+'.receipt.json',
                                        candidates, str(candidates)+'.receipt.json')
    result = {'split': 'val', 'manifest_sha256': sha256_file(manifest),
              'candidates_sha256': sha256_file(candidates), 'arms': {}}
    for name in ('transformer_v1', 'candidate_transformer_v1', 'candidate_motion_transformer_v2'):
        run = args.run_root / name
        predictions = run / 'validation_predictions.jsonl.gz'
        receipt = json.loads((run / 'receipt.json').read_text())
        if sha256_file(predictions) != receipt['validation_predictions_sha256']:
            raise ValueError('prediction hash mismatch')
        if receipt['validation_candidates_sha256'] != result['candidates_sha256']:
            raise ValueError('candidate provenance mismatch')
        result['arms'][name] = audit(aligned, list(read_jsonl(predictions)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
