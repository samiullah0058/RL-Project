from dataclasses import dataclass
from typing import Optional

from rlprompt.modules.ppo_module import PPOModule
from rlprompt.models import BaseModel
from rlprompt.rewards import BaseReward


def make_ppo_module(
    model: BaseModel,
    reward: BaseReward,
    config: "DictConfig",
) -> PPOModule:
    return PPOModule(
        model=model,
        reward=reward,
        clip_epsilon=config.clip_epsilon,
        value_loss_coef=config.value_loss_coef,
        ppo_epochs=config.ppo_epochs,
        reward_shaping=config.reward_shaping,
        reward_shaping_old_min=config.reward_shaping_old_min,
        reward_shaping_old_max=config.reward_shaping_old_max,
        reward_shaping_new_min=config.reward_shaping_new_min,
        reward_shaping_new_max=config.reward_shaping_new_max,
        reward_norm_decay=config.reward_norm_decay,
        reward_norm_eps=config.reward_norm_eps,
        top_k=config.top_k,
        top_p=config.top_p,
        num_beams=config.num_beams,
        racing_slice_fraction=config.racing_slice_fraction,
        racing_early_penalty=config.racing_early_penalty,
        racing_baseline_momentum=config.racing_baseline_momentum,
        enable_early_termination=config.enable_early_termination,
    )


@dataclass
class PPOModuleConfig:
    # PPO optimization
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    ppo_epochs: int = 4
    # EMA reward normalization
    reward_norm_decay: float = 0.99
    reward_norm_eps: float = 1e-8
    # Early-termination racing engine
    racing_slice_fraction: float = 0.25
    racing_early_penalty: float = -0.1
    racing_baseline_momentum: float = 0.9
    enable_early_termination: bool = True
    # Reward shaping linearly transforms reward range of [old_min, old_max]
    # to [new_min, new_max]
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    # Prompt generation setting
    top_k: Optional[int] = 50
    top_p: float = 1.0
    num_beams: int = 1
