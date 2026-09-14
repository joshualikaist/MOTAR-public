"""Frame-at-a-time perception; no annotation is accepted by the online selector."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from build_perception_motion_features import (
    DEFAULT_CONFIG, compute_flow, estimate_gmc, candidate_motion_features)
from perception_candidates import appearance_descriptor, sha256_file
from perception_temporal import candidate_feature, build_temporal_model
from perception_crop_verifier import CropVerifier, crop_tensor
from run_perception_candidate_producer import tile_origins


class FrozenDetector:
    def __init__(self, weights, expected_sha256, yolov5, device='cuda:0'):
        if sha256_file(weights) != expected_sha256:
            raise ValueError('detector hash mismatch')
        sys.path.insert(0, str(yolov5))
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression
        from torchvision.ops import nms
        self.device = torch.device(device)
        self.model = DetectMultiBackend(str(weights), device=self.device, fp16=False)
        self.model.warmup(imgsz=(8, 3, 640, 640))
        self.nms, self.tile_nms = nms, non_max_suppression

    @torch.no_grad()
    def __call__(self, frame):
        height, width = frame.shape[:2]
        if min(height, width) < 640:
            raise ValueError('frozen detector requires image dimensions >=640')
        origins = [(x, y) for y in tile_origins(height, 640, 128)
                   for x in tile_origins(width, 640, 128)]
        translated = []
        for start in range(0, len(origins), 8):
            batch = origins[start:start+8]
            tiles = [frame[y:y+640, x:x+640] for x, y in batch]
            array = np.ascontiguousarray(np.stack(tiles)[:, :, :, ::-1].transpose(0, 3, 1, 2))
            pred = self.model(torch.from_numpy(array).to(self.device).float()/255.)
            detections = self.tile_nms(pred, .001, .45, max_det=100)
            for det, (x, y) in zip(detections, batch):
                if len(det):
                    det = det[:, :5].detach().float().cpu()
                    det[:, [0, 2]] += x; det[:, [1, 3]] += y
                    translated.append(det)
        if not translated:
            return []
        merged = torch.cat(translated)
        merged[:, [0, 2]] = merged[:, [0, 2]].clamp(0., float(width))
        merged[:, [1, 3]] = merged[:, [1, 3]].clamp(0., float(height))
        selected = merged[self.nms(merged[:, :4], merged[:, 4], .45)[:5]].tolist()
        selected.sort(key=lambda x: -x[4])
        candidates = []
        for x1, y1, x2, y2, confidence in selected:
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append({'rank': len(candidates), 'u_px': (x1+x2)/2,
                               'v_px': (y1+y2)/2, 'width_px': x2-x1,
                               'height_px': y2-y1, 'confidence': confidence,
                               'appearance_64d': appearance_descriptor(frame, [x1,y1,x2,y2], 1.25)})
        return candidates


class StreamingSelector:
    def __init__(self, checkpoint, expected_sha256, device='cuda:0'):
        if sha256_file(checkpoint) != expected_sha256:
            raise ValueError('selector checkpoint hash mismatch')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.config = payload['config']
        self.device = torch.device(device)
        self.model = build_temporal_model(self.config).to(self.device).eval()
        self.model.load_state_dict(payload['model_state_dict'])
        self.history_length = self.config['history_length']
        self.reset()

    def reset(self):
        self.sequence = None
        self.frames = deque(maxlen=self.history_length)
        self.previous_gray = None
        self.previous_candidates = []
        self.previous_timestamp = None
        self.geometry = None

    @torch.no_grad()
    def step(self, candidates, sequence, timestamp_ns, width, height, motion):
        if sequence != self.sequence:
            self.reset(); self.sequence = sequence
        if self.previous_timestamp is not None and timestamp_ns <= self.previous_timestamp:
            raise ValueError('non-monotonic timestamp')
        if self.geometry is not None and self.geometry != (width, height):
            raise ValueError('geometry changed within sequence')
        features, mask = np.zeros((5,69),np.float32), np.zeros(5,bool)
        for i, c in enumerate(candidates):
            if c['rank'] != i or i >= 5:
                raise ValueError('invalid candidate ranks')
            features[i] = candidate_feature(c, width, height); mask[i] = True
        motion = np.asarray(motion, np.float32)
        if motion.shape != (5,12) or not np.isfinite(motion).all():
            raise ValueError('invalid motion features')
        self.frames.append((features, mask, timestamp_ns, motion))
        self.previous_timestamp, self.geometry = timestamp_ns, (width, height)
        t = self.history_length
        x, masks = np.zeros((1,t,5,69),np.float32), np.zeros((1,t,5),bool)
        frame_mask, delta, age = np.zeros((1,t),bool), np.zeros((1,t),np.float32), np.zeros((1,t),np.float32)
        motions = np.zeros((1,t,5,12),np.float32)
        for i, (f,m,ts,mot) in enumerate(self.frames):
            x[0,i], masks[0,i], motions[0,i], frame_mask[0,i] = f,m,mot,True
            age[0,i] = min((timestamp_ns-ts)/1e9,10.)
            if i:
                delta[0,i] = min((ts-self.frames[i-1][2])/1e9,1.)
        tensors = [torch.from_numpy(v).to(self.device) for v in (x,masks,frame_mask,delta)]
        logits = self.model(*tensors, torch.tensor([len(self.frames)],device=self.device),
                            age_seconds=torch.from_numpy(age).to(self.device),
                            motion_features=torch.from_numpy(motions).to(self.device))[0]
        values = logits.cpu().numpy()
        rank = int(values.argmax())
        return {'predicted_rank': rank if rank < 5 else None, 'logits': values.tolist()}


class PerceptionPipeline:
    def __init__(self, detector, selector, verifier=None, overlap_flow=True):
        self.detector, self.selector, self.verifier = detector, selector, verifier
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='perception-flow') if overlap_flow else None
        self._closed = False

    def close(self):
        if self._worker is not None:
            self._worker.shutdown(wait=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, frame, sequence, timestamp_ns, gray_frame=None):
        began = time.perf_counter()
        if self._closed:
            raise RuntimeError('pipeline is closed')
        s = self.selector
        if sequence != s.sequence:
            s.reset(); s.sequence = sequence
        if s.previous_timestamp is not None and timestamp_ns <= s.previous_timestamp:
            raise ValueError('non-monotonic timestamp')
        height, width = frame.shape[:2]
        if s.geometry is not None and s.geometry != (width,height):
            raise ValueError('geometry changed within sequence')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray_frame is None else gray_frame
        if gray.shape != (height,width):
            raise ValueError('gray geometry mismatch')
        scale = min(1.,640./width)
        if scale < 1:
            gray = cv2.resize(gray,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
        after_gray = time.perf_counter()
        future = None
        if self._worker is not None and s.previous_gray is not None:
            future = self._worker.submit(compute_flow, s.previous_gray, gray, DEFAULT_CONFIG)
        try:
            candidates = self.detector(frame)
        except BaseException:
            if future is not None:
                # Drain work before the caller can retry/reset the stream.
                try:
                    future.result()
                except Exception:
                    pass
            raise
        after_detector = time.perf_counter()
        motion = np.zeros((5,12),np.float32)
        if s.previous_gray is not None:
            flow = future.result() if future is not None else compute_flow(s.previous_gray, gray, DEFAULT_CONFIG)
            flow, affine, ratio, valid = estimate_gmc(flow,candidates,scale,DEFAULT_CONFIG)
            motion = candidate_motion_features(flow,affine,ratio,valid,candidates,
                       s.previous_candidates,width,height,scale,DEFAULT_CONFIG)
        after_motion = time.perf_counter()
        result = s.step(candidates,sequence,timestamp_ns,width,height,motion)
        s.previous_gray, s.previous_candidates = gray, candidates
        if self.verifier is not None:
            with torch.no_grad():
                scores = self.verifier(torch.from_numpy(crop_tensor(frame,candidates)).to(s.device))[0]
            scores[len(candidates):] = -torch.inf
            result['predicted_rank'] = int(scores.argmax()) if scores.max() >= 0 else None
        end = time.perf_counter()
        result['candidates'] = candidates
        result['motion_features'] = motion
        result['latency_ms'] = {'gray':(after_gray-began)*1000,
                                'detector':(after_detector-after_gray)*1000,
                                'motion':(after_motion-after_detector)*1000,
                                'selector':(end-after_motion)*1000,'total':(end-began)*1000}
        return result
