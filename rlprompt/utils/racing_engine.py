from typing import Any, Dict, List, Tuple

import torch

from rlprompt.rewards import BaseReward


class EarlyTerminationRacingEngine:
    """Screen rollouts on a data slice before committing to full reward evaluation."""

    def __init__(
        self,
        slice_fraction: float = 0.25,
        early_penalty: float = -0.1,
        baseline_momentum: float = 0.9,
        enabled: bool = True,
    ):
        self.slice_fraction = slice_fraction
        self.early_penalty = early_penalty
        self.baseline_momentum = baseline_momentum
        self.enabled = enabled
        self.baseline = None

    def slice_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        list_lengths = [len(v) for v in batch.values() if isinstance(v, list)]
        if len(list_lengths) == 0:
            return batch

        batch_size = list_lengths[0]
        slice_size = max(1, int(batch_size * self.slice_fraction))
        sliced_batch = {}
        for key, value in batch.items():
            if isinstance(value, list):
                sliced_batch[key] = value[:slice_size]
            else:
                sliced_batch[key] = value
        return sliced_batch

    def should_early_terminate(self, partial_score: float) -> bool:
        if not self.enabled:
            return False
        if self.baseline is None:
            return False
        return partial_score < self.baseline

    def update_baseline(self, reward: float) -> None:
        if self.baseline is None:
            self.baseline = reward
            return
        self.baseline = (
            self.baseline_momentum * self.baseline
            + (1.0 - self.baseline_momentum) * reward
        )

    def evaluate_rewards(
        self,
        reward_fn: BaseReward,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        mode: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        slice_batch = self.slice_batch(batch)
        rewards: List[float] = []
        early_terminated = 0
        aggregate_log: Dict[str, List[float]] = {}

        for tokens in output_tokens:
            prompt_tokens = [tokens]
            partial_reward, _ = reward_fn(
                **slice_batch,
                output_tokens=prompt_tokens,
                to_tensor=True,
                mode=mode,
            )
            partial_score = partial_reward[0].item()

            if self.should_early_terminate(partial_score):
                rewards.append(self.early_penalty)
                early_terminated += 1
                continue

            full_reward, reward_log = reward_fn(
                **batch,
                output_tokens=prompt_tokens,
                to_tensor=True,
                mode=mode,
            )
            reward_value = full_reward[0].item()
            rewards.append(reward_value)
            self.update_baseline(reward_value)

            for key, value in reward_log.items():
                aggregate_log.setdefault(key, []).append(float(value))

        if self.baseline is None and len(rewards) > 0:
            self.baseline = sum(rewards) / len(rewards)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        log = {
            key: sum(values) / len(values)
            for key, values in aggregate_log.items()
        }
        log["racing/early_terminated_fraction"] = (
            early_terminated / max(len(output_tokens), 1)
        )
        log["racing/baseline"] = self.baseline if self.baseline is not None else 0.0
        return rewards_tensor, log, early_terminated
