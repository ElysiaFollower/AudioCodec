"""Tests for the local gradient balancer."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from audiocodec.balancer import Balancer
else:
    Balancer = None


@unittest.skipIf(torch is None, "torch is not installed")
class BalancerTests(unittest.TestCase):
    def test_balancer_backward_accumulates_finite_gradients(self) -> None:
        reconstruction = torch.randn(2, 1, 128, requires_grad=True)
        target = torch.randn(2, 1, 128)

        losses = {
            "waveform_loss": torch.nn.functional.l1_loss(reconstruction, target),
            "stft_loss": ((reconstruction - target) ** 2).mean(),
        }
        balancer = Balancer(
            weights={"waveform_loss": 0.1, "stft_loss": 2.0},
            balance_grads=True,
            total_norm=1.0,
        )

        effective_loss = balancer.backward(losses, reconstruction)

        self.assertTrue(torch.isfinite(effective_loss))
        self.assertIsNotNone(reconstruction.grad)
        self.assertTrue(torch.isfinite(reconstruction.grad).all())

    def test_balancer_state_dict_round_trip_restores_ema_norms(self) -> None:
        reconstruction = torch.randn(2, 1, 128, requires_grad=True)
        target = torch.randn(2, 1, 128)
        losses = {
            "waveform_loss": torch.nn.functional.l1_loss(reconstruction, target),
            "stft_loss": ((reconstruction - target) ** 2).mean(),
        }

        balancer = Balancer(weights={"waveform_loss": 0.1, "stft_loss": 2.0})
        balancer.backward(losses, reconstruction)
        state = balancer.state_dict()

        restored = Balancer(weights={"waveform_loss": 0.1, "stft_loss": 2.0})
        restored.load_state_dict(state)

        self.assertEqual(set(restored._ema_norms), {"waveform_loss", "stft_loss"})


if __name__ == "__main__":
    unittest.main()
