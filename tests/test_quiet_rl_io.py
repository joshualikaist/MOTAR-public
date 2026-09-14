import importlib.util
from pathlib import Path
import unittest


_ROOT = Path(__file__).parents[1]
_PATH = _ROOT / "aerial_gym/rl_training/rl_games/quiet_rl_io.py"
_SPEC = importlib.util.spec_from_file_location("quiet_rl_io_standalone", _PATH)
_QUIET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_QUIET)


class QuietRLIOTest(unittest.TestCase):
    def test_critical_ppo_safety_messages_are_never_filtered(self):
        messages = (
            "[aerial RL] PPO epoch rejection latched before minibatch step",
            "[aerial RL] PPO EPOCH ROLLBACK | reason=KL",
            "[aerial RL] FAIL-STOP: same-density capture collapse",
            "[aerial RL] Resume optimizer LR reset to active config",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(_QUIET._allow_print(message))

    def test_unrelated_framework_chatter_remains_filtered(self):
        self.assertFalse(_QUIET._allow_print("saving next best rewards"))
        self.assertFalse(_QUIET._allow_print("unrelated debug tensor"))


if __name__ == "__main__":
    unittest.main()
