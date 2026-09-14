"""P7e UAV crop validity, with clip-fold model selection on NPS train only."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from perception_candidates import sha256_file
from perception_temporal import box_iou, candidate_box, load_aligned_records
from train_perception_temporal import seed_everything, association_metrics, git_value


def crop_tensor(frame, candidates):
    crops = np.zeros((5, 3, 32, 32), dtype=np.uint8)
    height, width = frame.shape[:2]
    for c in candidates:
        hw, hh = c['width_px'] * .625, c['height_px'] * .625
        x1, y1 = max(0, int(np.floor(c['u_px']-hw))), max(0, int(np.floor(c['v_px']-hh)))
        x2, y2 = min(width, int(np.ceil(c['u_px']+hw))), min(height, int(np.ceil(c['v_px']+hh)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('empty crop')
        patch = cv2.resize(frame[y1:y2, x1:x2], (32, 32), interpolation=cv2.INTER_AREA)
        crops[c['rank']] = patch[:, :, ::-1].transpose(2, 0, 1)
    return crops


class CropVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, 1)

    def forward(self, crops):
        embedding = self.encoder(crops.float()/255.).flatten(1)
        return self.head(embedding).squeeze(1), F.normalize(embedding, dim=1)


def load_crops(root, split):
    if split not in ('train', 'val'):
        raise ValueError('train/val only')
    manifest = root / ('nps_'+split+'_sequence.jsonl')
    candidates = root / ('nps_'+split+'_candidates.jsonl.gz')
    meta = json.loads(Path(str(manifest)+'.receipt.json').read_text())
    if meta['split'] != split:
        raise ValueError('split mismatch')
    aligned, _, receipt = load_aligned_records(
        manifest, str(manifest)+'.receipt.json', candidates, str(candidates)+'.receipt.json')
    arrays, labels, masks, clips = [], [], [], []
    for i, row in enumerate(aligned):
        src, cs = row['source'], row['candidate_record']['candidates']
        frame = cv2.imread(str(Path(meta['dataset'])/src['image']))
        if frame is None or frame.shape[:2] != (src['height_px'], src['width_px']):
            raise ValueError('image geometry mismatch')
        arrays.append(crop_tensor(frame, cs))
        target, mask = np.zeros(5, np.float32), np.zeros(5, bool)
        for c in cs:
            overlap = max([box_iou(candidate_box(c), gt) for gt in src['ground_truth_xyxy']] or [0.])
            target[c['rank']] = overlap >= .3
            mask[c['rank']] = overlap >= .3 or overlap < .1
        labels.append(target); masks.append(mask); clips.append(src['source_sequence_id'])
        if (i+1) % 2000 == 0:
            print('crop load', split, i+1, flush=True)
    return aligned, np.asarray(arrays), np.asarray(labels), np.asarray(masks), clips, receipt


def fit(crops, targets, train_ids, val_ids, epochs, device):
    seed_everything(17)
    model = CropVerifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.RandomState(17)
    history = []
    for epoch in range(epochs):
        model.train()
        ids = rng.permutation(train_ids)
        for offset in range(0, len(ids), 256):
            batch = ids[offset:offset+256]
            x = torch.from_numpy(crops[batch]).to(device)
            y = torch.from_numpy(targets[batch]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x)[0], y)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite crop loss')
            loss.backward(); opt.step()
        model.eval()
        total = 0.
        with torch.no_grad():
            for offset in range(0, len(val_ids), 256):
                batch = val_ids[offset:offset+256]
                logits = model(torch.from_numpy(crops[batch]).to(device))[0]
                total += F.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(targets[batch]).to(device), reduction='sum').item()
        history.append(total/len(val_ids) if len(val_ids) else None)
        print('crop epoch', epoch, 'heldout BCE', history[-1], flush=True)
    return model, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    source_commit = git_value('rev-parse', 'HEAD')
    source_hash = sha256_file(__file__)
    if a.output.exists():
        raise ValueError('refusing existing output')
    rows, crops, labels, masks, clips, train_receipt = load_crops(a.data_root, 'train')
    names = sorted(set(clips), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    folds = {name: i % 3 for i, name in enumerate(names)}
    fold_ids = np.repeat([folds[x] for x in clips], 5)
    x, y, valid = crops.reshape(-1, 3, 32, 32), labels.ravel(), masks.ravel()
    histories, best_epochs = [], []
    for fold in range(3):
        _, history = fit(x, y, np.flatnonzero(valid & (fold_ids != fold)),
                         np.flatnonzero(valid & (fold_ids == fold)), 8, a.device)
        histories.append(history); best_epochs.append(int(np.argmin(history))+1)
    epochs = int(np.median(best_epochs))
    model, _ = fit(x, y, np.flatnonzero(valid), np.asarray([], dtype=int), epochs, a.device)
    # Validation only becomes available after train-only epoch selection and full fit.
    val_rows, val_x, _, _, _, val_receipt = load_crops(a.data_root, 'val')
    if train_receipt['weights_sha256'] != val_receipt['weights_sha256']:
        raise ValueError('detector provenance differs')
    flat = val_x.reshape(-1, 3, 32, 32)
    scores = []
    with torch.no_grad():
        for start in range(0, len(flat), 256):
            scores.extend(model(torch.from_numpy(flat[start:start+256]).to(a.device))[0].cpu().tolist())
    scores = np.asarray(scores).reshape(-1, 5)
    predicted = []
    for row, logits in zip(val_rows, scores):
        logits[len(row['candidate_record']['candidates']):] = -np.inf
        predicted.append(int(logits.argmax()) if logits.max() >= 0 else 5)
    metrics, _ = association_metrics(type('Rows', (), {'records': val_rows})(), predicted, .3)
    a.output.mkdir(parents=True)
    torch.save({'model_state_dict': model.cpu().state_dict(), 'epochs': epochs,
                'detector_weights_sha256': train_receipt['weights_sha256'],
                'crop_contract': 'BGR to RGB, context1.25,32x32,INTER_AREA'}, a.output/'best.pt')
    report = {'folds': folds, 'fold_bce': histories, 'fold_best_epochs': best_epochs,
              'epochs': epochs, 'metrics': metrics,
              'train_candidates_sha256': train_receipt['output_sha256'],
              'val_candidates_sha256': val_receipt['output_sha256'],
              'source_git_commit': source_commit, 'source_sha256': source_hash,
              'checkpoint_sha256': sha256_file(a.output/'best.pt'),
              'validation_predicted_rank': predicted, 'test_used': False}
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('P7e utility', metrics['selection_utility'], flush=True)


if __name__ == '__main__':
    main()
