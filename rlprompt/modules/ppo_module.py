import torch
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import get_reward_shaping_func
from rlprompt.losses.ppo_losses import ppo_loss_with_sparse_rewards
from rlprompt.utils.reward_normalization import EMARewardNormalizer
from rlprompt.utils.racing_engine import EarlyTerminationRacingEngine
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOModule(BaseModule):
    """Actor-Critic PPO module for discrete prompt optimization."""

    def __init__(
        self,
        model: BaseModel,
        reward: BaseReward,
        clip_epsilon: float,
        value_loss_coef: float,
        ppo_epochs: int,
        reward_shaping: bool,
        reward_shaping_old_min: float,
        reward_shaping_old_max: float,
        reward_shaping_new_min: float,
        reward_shaping_new_max: float,
        reward_norm_decay: float,
        reward_norm_eps: float,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        racing_slice_fraction: float,
        racing_early_penalty: float,
        racing_baseline_momentum: float,
        enable_early_termination: bool,
    ):
        super().__init__()
        assert not (top_k is not None and top_p < 1.0), \
            "Only one of top_k or top_p should be selected"

        self._model = model
        self._reward = reward
        self._clip_epsilon = clip_epsilon
        self._value_loss_coef = value_loss_coef
        self._ppo_epochs = ppo_epochs
        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams

        if reward_shaping is True:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r

        self._reward_normalizer = EMARewardNormalizer(
            decay=reward_norm_decay,
            eps=reward_norm_eps,
        )
        self._racing_engine = EarlyTerminationRacingEngine(
            slice_fraction=racing_slice_fraction,
            early_penalty=racing_early_penalty,
            baseline_momentum=racing_baseline_momentum,
            enabled=enable_early_termination,
        )

        self._rollout_buffer: Dict[str, torch.Tensor] = {}

    def get_policy_model(self) -> BaseModel:
        return self._model

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._collect_rollout(batch)

        loss_list = []
        loss_log_list = []
        for _ in range(self._ppo_epochs):
            loss, loss_log = self._ppo_update(batch)
            loss_list.append(loss)
            loss_log_list.append(loss_log)

        total_loss = torch.mean(torch.stack(loss_list))
        loss_log = utils.unionize_dicts(loss_log_list)
        return total_loss, loss_log

    def _collect_rollout(self, batch: Dict[str, Any]) -> None:
        was_training = self._model.training
        self._model.eval()
        with torch.no_grad():
            outputs = self._decode_rollout(batch=batch)
            raw_rewards, rewards_log, _ = self._compute_rewards_with_racing(
                batch=batch,
                output_tokens=outputs['sample_tokens'],
                mode="train",
            )
            shaped_rewards = self._reward_shaping_func(raw_rewards)
            normalized_rewards = self._reward_normalizer.normalize_and_update(
                shaped_rewards)

            self._rollout_buffer = {
                "sample_ids": outputs['sample_ids'].detach(),
                "old_log_probs": outputs['sample_log_probs'].detach(),
                "old_values": outputs['sample_values'].detach(),
                "sequence_lengths": outputs['sample_lengths'].detach(),
                "rewards": normalized_rewards.detach(),
                "rewards_log": rewards_log,
                "raw_reward_mean": raw_rewards.mean().detach(),
                "shaped_reward_mean": shaped_rewards.mean().detach(),
            }
        if was_training:
            self._model.train()

    def _ppo_update(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_ = {k: v for k, v in batch.items()}
        batch_.update({
            "sample_ids": self._rollout_buffer["sample_ids"],
        })
        outputs = self._model.teacher_forcing(
            **batch_,
            top_k=self._top_k,
            top_p=self._top_p,
        )

        ppo_loss, ppo_loss_log = ppo_loss_with_sparse_rewards(
            new_log_probs=outputs['sample_log_probs'],
            old_log_probs=self._rollout_buffer["old_log_probs"],
            values=outputs['sample_values'],
            rewards=self._rollout_buffer["rewards"],
            sequence_lengths=self._rollout_buffer["sequence_lengths"],
            clip_epsilon=self._clip_epsilon,
            value_loss_coef=self._value_loss_coef,
        )

        rewards_log = self._rollout_buffer["rewards_log"]
        prefixed_reward_log = {
            f"ppo/rewards/{key}": torch.tensor(value)
            for key, value in rewards_log.items()
        }
        ppo_loss_log = utils.unionize_dicts([
            prefixed_reward_log,
            ppo_loss_log,
            {
                "ppo/rewards/raw": self._rollout_buffer["raw_reward_mean"],
                "ppo/rewards/shaped": self._rollout_buffer["shaped_reward_mean"],
                "ppo/rewards/normalized": self._rollout_buffer["rewards"].mean(),
                "ppo/reward_norm/mean": torch.tensor(
                    self._reward_normalizer.mean),
                "ppo/reward_norm/std": torch.tensor(
                    (self._reward_normalizer.var + self._reward_normalizer.eps)
                    ** 0.5),
            },
        ])
        return ppo_loss, ppo_loss_log

    def _compute_rewards_with_racing(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        rewards_tensor, rewards_log, early_terminated = \
            self._racing_engine.evaluate_rewards(
                reward_fn=self._reward,
                batch=batch,
                output_tokens=output_tokens,
                mode=mode,
            )
        return rewards_tensor, rewards_log, early_terminated

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if mode != "train":
            rewards_tensor, rewards_log = self._reward(
                **batch,
                output_tokens=output_tokens,
                to_tensor=to_tensor,
                mode=mode,
            )
            rewards_tensor = rewards_tensor.to(device)
            return rewards_tensor, rewards_log

        rewards_tensor, rewards_log, _ = self._compute_rewards_with_racing(
            batch=batch,
            output_tokens=output_tokens,
            mode=mode,
        )
        return rewards_tensor, rewards_log

    def infer(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        return self._model.generate(
            **batch,
            do_sample=False,
            top_k=self._top_k,
            top_p=self._top_p,
            num_beams=self._num_beams,
            infer=True,
        )

    def _decode_rollout(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._model.generate(
            **batch,
            do_sample=True,
            top_k=self._top_k,
            top_p=self._top_p,
            num_beams=self._num_beams,
        )
