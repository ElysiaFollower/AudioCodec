"""Local gradient balancer adapted for single-process AudioCodec training."""

from __future__ import annotations

import torch
from torch import autograd


class Balancer:
    """Balance gradient contributions from multiple losses on a shared tensor."""

    def __init__(
        self,
        weights: dict[str, float],
        balance_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        if not weights:
            raise ValueError("weights must not be empty.")
        if sum(weights.values()) <= 0:
            raise ValueError("weights must sum to a positive value.")
        self.weights = weights
        self.balance_grads = balance_grads
        self.total_norm = total_norm
        self.ema_decay = ema_decay
        self.per_batch_item = per_batch_item
        self.epsilon = epsilon
        self._ema_norms: dict[str, torch.Tensor] = {}
        self.metrics: dict[str, float] = {}

    def state_dict(self) -> dict:
        return {
            "ema_norms": {name: value.clone() for name, value in self._ema_norms.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        ema_norms = state_dict.get("ema_norms", {})
        self._ema_norms = {name: value.clone() for name, value in ema_norms.items()}
        self.metrics = {}

    def backward(self, losses: dict[str, torch.Tensor], input_tensor: torch.Tensor) -> torch.Tensor:
        if not losses:
            raise ValueError("losses must not be empty.")
        missing = set(losses) - set(self.weights)
        if missing:
            raise ValueError(f"weights are missing keys for losses: {sorted(missing)}")

        grads: dict[str, torch.Tensor] = {}
        norms: dict[str, torch.Tensor] = {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [input_tensor], retain_graph=True)
            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
            grads[name] = grad
            norms[name] = norm.detach()

        averaged_norms: dict[str, torch.Tensor] = {}
        for name, norm in norms.items():
            previous = self._ema_norms.get(name)
            if previous is None:
                averaged = norm
            else:
                averaged = self.ema_decay * previous + (1.0 - self.ema_decay) * norm
            self._ema_norms[name] = averaged.detach()
            averaged_norms[name] = averaged

        total_weight = sum(self.weights[name] for name in losses)
        desired_ratios = {name: self.weights[name] / total_weight for name in losses}

        output_grad = torch.zeros_like(input_tensor)
        effective_loss = torch.zeros((), device=input_tensor.device, dtype=input_tensor.dtype)
        self.metrics = {}
        for name, grad in grads.items():
            avg_norm = averaged_norms[name]
            if self.balance_grads:
                scale = desired_ratios[name] * self.total_norm / (self.epsilon + avg_norm)
            else:
                scale = input_tensor.new_tensor(self.weights[name])
            output_grad.add_(grad, alpha=float(scale))
            effective_loss = effective_loss + scale * losses[name].detach()
            self.metrics[f"ratio_{name}"] = float(desired_ratios[name])

        input_tensor.backward(output_grad)
        return effective_loss
