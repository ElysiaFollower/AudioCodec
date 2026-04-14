"""Configuration tests."""

from pathlib import Path
import unittest

from audiocodec.config import load_experiment_config


class ConfigTests(unittest.TestCase):
    def test_baseline_config_exposes_dataset_root(self) -> None:
        config = load_experiment_config(Path("configs/baseline.json"))
        self.assertEqual(
            config.dataset.root,
            "/home/lujingyu/lujingyu_data/data/AUDIO_DATA/librispeech_asr/clean/train.100",
        )
        self.assertEqual(config.frame_rate, 100)
        self.assertEqual(config.model.architecture, "conv")
        self.assertFalse(config.adversarial.enabled)

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
        self.assertFalse(config.optimization.mixed_precision)
        self.assertFalse(config.adversarial.enabled)

    def test_encodec_like_loss_ablation_matches_reference_direction(self) -> None:
        config = load_experiment_config(Path("configs/ablation-encodec-like-loss.json"))
        self.assertEqual(config.model.architecture, "seanet")
        self.assertEqual(config.loss.waveform_weight, 0.1)
        self.assertEqual(config.loss.stft_weight, 2.0)
        self.assertEqual(config.loss.mel_weight, 0.0)
        self.assertFalse(config.optimization.mixed_precision)
        self.assertFalse(config.adversarial.enabled)

    def test_adversarial_ablation_config_enables_msstft_discriminator(self) -> None:
        config = load_experiment_config(Path("configs/ablation-adversarial-msstft.json"))
        self.assertTrue(config.adversarial.enabled)
        self.assertEqual(config.adversarial.discriminator, "msstftd")
        self.assertEqual(config.adversarial.loss_type, "hinge")
        self.assertEqual(config.adversarial.adversarial_weight, 4.0)
        self.assertEqual(config.adversarial.feature_matching_weight, 4.0)


if __name__ == "__main__":
    unittest.main()
