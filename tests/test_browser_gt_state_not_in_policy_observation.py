"""Browser GT preview fields must never enter the actor observation tensor."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BrowserGtObservationBoundaryTest(unittest.TestCase):
    def test_browser_gt_state_not_in_policy_observation(self):
        forbidden = (
            'gt-route-track',
            'gt-free-roam',
            'gt_route_track',
            'predictedFollowPoint',
            'NavRLArenaDemoPlanner',
            'BROWSER GT PREVIEW',
        )
        observation_files = [
            ROOT / 'aerial_gym/task/navrl_task/navrl_task.py',
            ROOT / 'aerial_gym/task/navrl_task/target_motion.py',
            ROOT / 'aerial_gym/config/task_config/navrl_task_config.py',
        ]
        for path in observation_files:
            text = path.read_text()
            for token in forbidden:
                self.assertNotIn(
                    token, text,
                    f'{token} leaked into {path.relative_to(ROOT)}',
                )
        task = (ROOT / 'aerial_gym/task/navrl_task/navrl_task.py').read_text()
        builder = task.split('def process_obs_for_task', 1)[1].split(
            '\n    def ', 1
        )[0]
        for token in (
            'browser_gt', 'exact target heading', 'predicted_follow',
            'gt-free-roam', 'gt-route-track',
        ):
            self.assertNotIn(token, builder)


if __name__ == '__main__':
    unittest.main()
