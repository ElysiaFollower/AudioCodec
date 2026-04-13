"""Model shape tests."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from audiocodec.config import load_experiment_config

if torch is not None:
    from audiocodec.models.codec import ConvRVQCodec
else:
    ConvRVQCodec = None


@unittest.skipIf(torch is None, "torch is not installed")
class ConvRVQCodecTests(unittest.TestCase):
    def test_forward_preserves_original_length(self) -> None:
        config = load_experiment_config("configs/baseline.json")
        model = ConvRVQCodec.from_config(config)

        waveform = torch.randn(2, config.audio.channels, 32_123)
        output = model(waveform)

        self.assertEqual(output.reconstruction.shape, waveform.shape)
        self.assertEqual(output.codes.shape[0], 2)
        self.assertEqual(output.codes.shape[1], config.quantizer.num_quantizers)
        self.assertEqual(output.codes.shape[2], output.padded_length // model.hop_length)

    def test_encode_then_decode_round_trips_shape(self) -> None:
        config = load_experiment_config("configs/baseline.json")
        model = ConvRVQCodec.from_config(config)

        waveform = torch.randn(1, config.audio.channels, 48_000)
        encoded = model.encode(waveform)
        reconstruction = model.decode(encoded.codes, length=encoded.original_length)

        self.assertEqual(reconstruction.shape, waveform.shape)


if __name__ == "__main__":
    unittest.main()
