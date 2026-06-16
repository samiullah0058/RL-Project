from typing import Dict, Tuple

import torch

from .loss_utils import mask_and_reduce


def ppo_loss_with_sparse_rewards(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    sequence_lengths: torch.LongTensor,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    normalize_advantages: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO-Clip policy loss with value-function MSE for sparse terminal rewards."""
    batch_size, seq_len = new_log_probs.shape
    returns = rewards.unsqueeze(1).expand(batch_size, seq_len)
    advantages = (returns - values.detach()).detach()

    if normalize_advantages:
        adv_mean = mask_and_reduce(
            advantages,
            sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False,
        )
        centered = advantages - adv_mean
        adv_var = mask_and_reduce(
            centered.pow(2),
            sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False,
        )
        advantages = centered / (adv_var.sqrt() + 1e-8)

    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -mask_and_reduce(
        torch.min(unclipped, clipped),
        sequence_lengths,
        average_across_timesteps=True,
        sum_over_timesteps=False,
    )

    value_loss = mask_and_reduce(
        (values - returns).pow(2),
        sequence_lengths,
        average_across_timesteps=True,
        sum_over_timesteps=False,
    )

    total_loss = policy_loss + value_loss_coef * value_loss

    with torch.no_grad():
        approx_kl = mask_and_reduce(
            old_log_probs - new_log_probs,
            sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False,
        )
        clip_fraction = mask_and_reduce(
            ((ratio - 1.0).abs() > clip_epsilon).float(),
            sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False,
        )

    quantities_to_log = {
        "ppo/policy_loss": policy_loss.detach(),
        "ppo/value_loss": value_loss.detach(),
        "ppo/total_loss": total_loss.detach(),
        "ppo/approx_kl": approx_kl,
        "ppo/clip_fraction": clip_fraction,
        "ppo/mean_return": rewards.mean().detach(),
        "ppo/mean_value": mask_and_reduce(
            values,
            sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False,
        ).detach(),
    }
    return total_loss, quantities_to_log
