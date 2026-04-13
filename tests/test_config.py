"""Configuration tests."""

from pathlib import Path
import unittest

from audiocodec.config import load_experiment_config


class ConfigTests(unittest.TestCase):
    def test_baseline_config_exposes_dataset_root(self) -> None:
        config = load_experiment_config(Path("configs/baseline.json"))
        self.assertEqual(config.dataset.root, "/path/to/LibriSpeech/dev-clean")
        self.assertEqual(config.frame_rate, 100)

    def test_mel_ablation_config_enables_mel_loss(self) -> None:
        config = load_experiment_config(Path("configs/ablation-mel-loss.json"))
        self.assertEqual(config.loss.mel_weight, 0.5)
        self.assertEqual(config.loss.mel_n_mels, 80)


if __name__ == "__main__":
    unittest.main()
