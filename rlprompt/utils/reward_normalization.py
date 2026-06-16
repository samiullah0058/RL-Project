import torch
from typing import Union


class EMARewardNormalizer:
    """Tracks a rolling mean and variance of rewards via exponential moving average."""

    def __init__(
        self,
        decay: float = 0.99,
        eps: float = 1e-8,
    ):
        self.decay = decay
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, rewards: torch.Tensor) -> None:
        batch_mean = rewards.detach().mean().item()
        batch_var = rewards.detach().var(unbiased=False).item()

        if not self.initialized:
            self.mean = batch_mean
            self.var = max(batch_var, self.eps)
            self.initialized = True
            return

        self.mean = self.decay * self.mean + (1.0 - self.decay) * batch_mean
        self.var = self.decay * self.var + (1.0 - self.decay) * batch_var

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            return rewards
        std = (self.var + self.eps) ** 0.5
        return (rewards - self.mean) / std

    def normalize_and_update(self, rewards: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(rewards)
        self.update(rewards)
        return normalized

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "initialized": self.initialized,
        }
