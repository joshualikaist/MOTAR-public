import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from threading import Event

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from perception_temporal import build_temporal_model
from perception_candidates import sha256_file
from perception_streaming import StreamingSelector, PerceptionPipeline
from build_perception_motion_features import compute_flow, estimate_gmc, estimate_backward_flow_and_gmc, DEFAULT_CONFIG
from perception_crop_verifier import crop_tensor, CropVerifier


class StreamingTest(unittest.TestCase):
    def test_flow_split_and_worker_are_byte_identical(self):
        from concurrent.futures import ThreadPoolExecutor
        previous=np.random.default_rng(17).integers(0,256,(64,80),dtype=np.uint8)
        current=np.roll(previous,2,axis=1)
        expected=estimate_backward_flow_and_gmc(previous,current,[],1.,DEFAULT_CONFIG)
        with ThreadPoolExecutor(max_workers=1) as worker:
            flow=worker.submit(compute_flow,previous,current).result()
        actual=estimate_gmc(flow,[],1.)
        for left,right in zip(expected[:2],actual[:2]):
            self.assertEqual(left.tobytes(),right.tobytes())
        self.assertEqual(expected[2:],actual[2:])

    def test_worker_overlaps_detector_and_close_prevents_reuse(self):
        from types import SimpleNamespace
        entered,detected=Event(),Event()
        gray=np.zeros((64,80),np.uint8)
        def flow(*args):
            entered.set()
            if not detected.wait(5): raise AssertionError('detector never overlapped flow')
            return np.zeros((64,80,2),np.float32)
        def detector(frame):
            if not entered.wait(5): raise AssertionError('flow was not submitted before detector')
            detected.set()
            return []
        selector=SimpleNamespace(sequence='a',previous_timestamp=0,geometry=(80,64),
            previous_gray=gray,previous_candidates=[],step=lambda *args: {'predicted_rank':None})
        with patch('perception_streaming.compute_flow',side_effect=flow):
            with PerceptionPipeline(detector,selector) as pipeline:
                result=pipeline.process(np.zeros((64,80,3),np.uint8),'a',1)
                self.assertEqual(result['motion_features'].tobytes(),np.zeros((5,12),np.float32).tobytes())
        with self.assertRaisesRegex(RuntimeError,'closed'):
            pipeline.process(np.zeros((64,80,3),np.uint8),'a',2)

    def test_sequence_reset_timestamp_guard_and_empty_frame(self):
        config=json.loads((ROOT/'configs/perception_candidate_motion_transformer_v1.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save({'config':config,'model_state_dict':build_temporal_model(config).state_dict()},path)
            with self.assertRaisesRegex(ValueError,'hash'):
                StreamingSelector(path,'0'*64,'cpu')
            selector=StreamingSelector(path,sha256_file(path),'cpu')
            zero=np.zeros((5,12),np.float32)
            result=selector.step([],'a',0,100,100,zero)
            self.assertIsNone(result['predicted_rank'])
            with self.assertRaisesRegex(ValueError,'timestamp'):
                selector.step([],'a',0,100,100,zero)
            with self.assertRaisesRegex(ValueError,'geometry'):
                selector.step([],'a',1,101,100,zero)
            selector.step([],'b',0,200,100,zero)
            self.assertEqual(len(selector.frames),1)

    def test_crop_rgb_channel_order_and_feature_norm(self):
        frame=np.zeros((40,40,3),np.uint8);frame[:,:,2]=255
        c={'rank':0,'u_px':20.,'v_px':20.,'width_px':10.,'height_px':10.}
        crops=crop_tensor(frame,[c])
        self.assertTrue((crops[0,0]==255).all())
        self.assertTrue((crops[0,2]==0).all())
        scores,features=CropVerifier()(torch.from_numpy(crops[:1]))
        self.assertEqual(tuple(scores.shape),(1,))
        self.assertAlmostEqual(float(features.norm()),1.,places=5)


if __name__=='__main__':unittest.main()
