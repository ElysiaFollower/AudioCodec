"""Tests for adversarial training utilities."""

import unittest

try:
    import torch
    import torchaudio  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from audiocodec.adversarial import (
        FeatureMatchingLoss,
        MultiScaleSTFTDiscriminator,
        discriminator_adversarial_loss,
        generator_adversarial_loss,
    )
else:
    FeatureMatchingLoss = None
    MultiScaleSTFTDiscriminator = None
    discriminator_adversarial_loss = None
    generator_adversarial_loss = None


@unittest.skipIf(torch is None, "torch and torchaudio are required")
class AdversarialTests(unittest.TestCase):
    def test_msstft_discriminator_returns_expected_nested_structure(self) -> None:
        discriminator = MultiScaleSTFTDiscriminator(
            filters=8,
            n_ffts=(512, 1024),
            hop_lengths=(128, 256),
            win_lengths=(512, 1024),
        )
        waveform = torch.randn(2, 1, 4096)

        logits, feature_maps = discriminator(waveform)

        self.assertEqual(len(logits), 2)
        self.assertEqual(len(feature_maps), 2)
        self.assertTrue(all(isinstance(scale_logits, torch.Tensor) for scale_logits in logits))
        self.assertTrue(all(isinstance(scale_maps, list) for scale_maps in feature_maps))
        self.assertTrue(all(len(scale_maps) > 0 for scale_maps in feature_maps))

    def test_hinge_and_feature_matching_losses_are_finite(self) -> None:
        discriminator = MultiScaleSTFTDiscriminator(
            filters=8,
            n_ffts=(512, 1024),
            hop_lengths=(128, 256),
            win_lengths=(512, 1024),
        )
        real = torch.randn(2, 1, 4096)
        fake = torch.randn(2, 1, 4096)

        fake_logits, fake_feature_maps = discriminator(fake)
        real_logits, real_feature_maps = discriminator(real)

        d_loss = discriminator_adversarial_loss(fake_logits, real_logits, loss_type="hinge")
        g_loss = generator_adversarial_loss(fake_logits, loss_type="hinge")
        fm_loss = FeatureMatchingLoss()(fake_feature_maps, real_feature_maps)

        self.assertTrue(torch.isfinite(d_loss))
        self.assertTrue(torch.isfinite(g_loss))
        self.assertTrue(torch.isfinite(fm_loss))


if __name__ == "__main__":
    unittest.main()
