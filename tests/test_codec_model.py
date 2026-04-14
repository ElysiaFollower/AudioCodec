"""Model shape tests."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from audiocodec.config import load_experiment_config

if torch is not None:
    from audiocodec.models.codec import build_codec_model
else:
    build_codec_model = None


@unittest.skipIf(torch is None, "torch is not installed")
class ConvRVQCodecTests(unittest.TestCase):
    def _assert_round_trip(self, config_path: str, waveform_length: int) -> None:
        config = load_experiment_config(config_path)
        model = build_codec_model(config)

        waveform = torch.randn(2, config.audio.channels, waveform_length)
        output = model(waveform)
        encoded = model.encode(waveform[:1])
        decoded = model.decode(encoded.codes, length=encoded.original_length)

        self.assertEqual(output.reconstruction.shape, waveform.shape)
        self.assertEqual(output.codes.shape[0], 2)
        self.assertEqual(output.codes.shape[1], config.quantizer.num_quantizers)
        self.assertEqual(output.codes.shape[2], output.padded_length // model.hop_length)
        self.assertEqual(decoded.shape, waveform[:1].shape)

    def test_conv_baseline_round_trip(self) -> None:
        self._assert_round_trip("configs/baseline.json", waveform_length=32_123)

    def test_seanet_round_trip(self) -> None:
        self._assert_round_trip("configs/encodec-inspired.json", waveform_length=48_321)


if __name__ == "__main__":
    unittest.main()
