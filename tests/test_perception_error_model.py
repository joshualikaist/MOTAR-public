import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_perception_error_model import size_bin, segments, summarize, transition_summary, STATES


def frame(index, state='HIT', size=10., gt_count=1, sequence='a', timestamp=None):
    return {'frame_id': str(index), 'sequence': sequence,
            'timestamp_ns': int(index * 100000000 if timestamp is None else timestamp),
            'conditional_eligible': gt_count == 1, 'gt_count': gt_count,
            'size_bin': size_bin(size) if gt_count == 1 else None,
            'state': state, 'width_px': 1280, 'height_px': 960,
            'center_offset_px': None if state == 'NO_LOCK' else [1., -2.],
            'center_offset_normalized': None if state == 'NO_LOCK' else [1./1280, -2./960],
            'latency_ms_repeats': [40., 41., 42.]}


class ErrorModelTest(unittest.TestCase):
    def test_fixed_bins_and_invalid_geometry(self):
        self.assertEqual([size_bin(x) for x in (7.99, 8., 16., 32., 64.)], list(range(5)))
        for x in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): size_bin(x)

    def test_exclusion_sequence_and_gap_break_transitions(self):
        frames = [frame(0), frame(1, gt_count=2), frame(2),
                  frame(3, sequence='b'), frame(4, sequence='b', timestamp=1000000000)]
        self.assertEqual(len(segments(frames)), 4)
        model, _ = summarize(frames)
        self.assertEqual(model['eligible_frames'], 4)
        self.assertEqual(model['excluded_gt_count_histogram'], {'2': 1})
        self.assertEqual(sum(map(sum,model['bins'][1]['transition_3state']['counts'])), 0)

    def test_loss_combines_false_and_no_lock_with_censoring(self):
        frames = [frame(0), frame(1, 'FALSE_LOCK'), frame(2, 'NO_LOCK'), frame(3), frame(4, 'NO_LOCK')]
        model, bursts = summarize(frames)
        losses = [b for b in bursts if b['state'] == 'LOST']
        self.assertEqual(len(losses), 2)
        self.assertEqual(losses[0]['duration_seconds'], .2)
        self.assertTrue(losses[0]['is_reacquisition'])
        self.assertTrue(losses[1]['right_censored'])
        self.assertFalse(losses[1]['is_reacquisition'])
        self.assertEqual(model['bins'][1]['state_counts'], {'HIT':2,'FALSE_LOCK':1,'NO_LOCK':2})
        self.assertEqual(model['bins'][1]['error']['HIT']['joint_du_dv_px_samples'], [[1.,-2.],[1.,-2.]])

    def test_size_change_attributes_transition_to_origin(self):
        model, _ = summarize([frame(0,size=7),frame(1,'FALSE_LOCK',size=17)])
        self.assertEqual(model['bins'][0]['transition_3state']['counts'][0][1], 1)
        self.assertIsNone(model['bins'][2]['transition_3state']['probabilities'][1])
        self.assertFalse(model['bins'][0]['supported'])
        self.assertIsNone(model['bins'][4]['latency_with_decode_ms']['mean'])

    def test_probabilities_support_and_interval_are_observational(self):
        result = transition_summary([('HIT','NO_LOCK',.1)]*10 + [('NO_LOCK','HIT',.2)], STATES)
        self.assertEqual(result['row_supported'], [True,False,False])
        self.assertEqual(result['probabilities'][0], [0.,0.,1.])
        self.assertEqual(result['interval_seconds']['n'], 11)

    def test_left_censored_loss_is_not_reacquisition(self):
        model, bursts = summarize([frame(0,'NO_LOCK'),frame(1)])
        self.assertEqual(model['reacquisition_seconds']['n'], 0)
        self.assertTrue(next(b for b in bursts if b['state']=='LOST')['left_censored'])


if __name__ == '__main__': unittest.main()
