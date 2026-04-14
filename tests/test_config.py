"""Configuration tests."""

from pathlib import Path
import unittest

from audiocodec.config import load_experiment_config


class ConfigTests(unittest.TestCase):
    def test_baseline_config_exposes_dataset_root(self) -> None:
        config = load_experiment_config(Path("configs/baseline.json"))
        self.assertEqual(config.dataset.root, "/path/to/LibriSpeech/dev-clean")
        self.assertEqual(config.frame_rate, 100)
        self.assertEqual(config.model.architecture, "conv")

    def test_mel_ablation_config_enables_mel_loss(self) -> None:
        config = load_experiment_config(Path("configs/ablation-mel-loss.json"))
        self.assertEqual(config.loss.mel_weight, 0.5)
        self.assertEqual(config.loss.mel_n_mels, 80)

    def test_encodec_inspired_config_uses_seanet_hop_length(self) -> None:
        config = load_experiment_config(Path("configs/encodec-inspired.json"))
        self.assertEqual(config.model.architecture, "seanet")
        self.assertEqual(config.frame_rate, 50)
        self.assertEqual(config.quantizer.codebook_size, 1024)
        self.assertEqual(config.quantizer.num_quantizers, 24)


if __name__ == "__main__":
    unittest.main()
