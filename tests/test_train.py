"""Tests for training utilities."""

from pathlib import Path
import tempfile
import unittest

from audiocodec.train import _assert_resume_config_matches, _read_logged_steps


class TrainingUtilityTests(unittest.TestCase):
    def test_read_logged_steps_tracks_last_step_and_best_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(
                    [
                        '{"split":"train","step":1,"total_loss":3.0}',
                        '{"split":"val","step":1000,"total_loss":2.5}',
                        '{"split":"train","step":1050,"total_loss":2.0}',
                        '{"split":"val","step":2000,"total_loss":1.5}',
                    ]
                )
            )

            last_step, best_val_loss = _read_logged_steps(metrics_path)

            self.assertEqual(last_step, 2000)
            self.assertEqual(best_val_loss, 1.5)

    def test_resume_config_mismatch_raises(self) -> None:
        current = {"model": {"architecture": "seanet"}, "optimization": {"batch_size": 8}}
        checkpoint = {"model": {"architecture": "conv"}, "optimization": {"batch_size": 8}}

        with self.assertRaises(ValueError):
            _assert_resume_config_matches(current, checkpoint)


if __name__ == "__main__":
    unittest.main()
