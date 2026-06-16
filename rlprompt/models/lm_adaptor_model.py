import torch
from torch import nn
import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer

from .base_model import BaseModel
from .model_utils import _top_k_logits, _top_p_logits


SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']

LM_HIDDEN_SIZES = {'distilgpt2': 768,
                   'gpt2': 768,
                   'gpt2-medium': 1024,
                   'gpt2-large': 1280,
                   'gpt2-xl': 1600}


class LMAdaptorModel(BaseModel):
    """Uses an MLP adaptor with disjoint Actor and Critic heads.

    The frozen backbone LM provides hidden states. A parameter-efficient
  bottleneck MLP feeds:
    - Actor head: adaptor output -> frozen lm_head -> token logits
    - Critic head: bottleneck hidden state -> scalar state value
    """

    def __init__(
        self,
        policy_lm: str,
        hidden_size: int,
        logit_bias: float,
        fluent: bool,
        fluent_top_k: Optional[int],
        max_decoding_length: int,
        eos_token_id: Optional[int]
    ):
        super().__init__()

        assert policy_lm in SUPPORTED_LMS
        model = policy_lm
        self.device = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False

        self.logit_bias = logit_bias
        self.fluent = fluent
        self.fluent_top_k = fluent_top_k
        self.max_decoding_length = max_decoding_length
        self.eos_token_id = eos_token_id

        model_dim = LM_HIDDEN_SIZES[model]
        self.actor_fc1 = nn.Linear(model_dim, hidden_size).to(self.device)
        self.actor_relu = nn.ReLU()
        self.actor_fc2 = nn.Linear(hidden_size, model_dim).to(self.device)
        self.critic_head = nn.Linear(hidden_size, 1).to(self.device)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.actor_fc1.apply(_init_weights)
        self.actor_fc2.apply(_init_weights)
        self.critic_head.apply(_init_weights)

    @property
    def mlp(self) -> nn.Sequential:
        """Backward-compatible accessor for the adaptor trunk."""
        return nn.Sequential(self.actor_fc1, self.actor_relu, self.actor_fc2)

    def _policy_forward(
        self,
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bottleneck = self.actor_relu(self.actor_fc1(state))
        adaptor_output = self.actor_fc2(bottleneck)
        logits = self.generator.model.lm_head(adaptor_output)
        value = self.critic_head(bottleneck).squeeze(-1)

        if self.fluent:
            lm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        return dict(logits=logits, value=value, bottleneck=bottleneck)

    def _mlp_forward(self, state: torch.Tensor) -> torch.Tensor:
        return self._policy_forward(state)['logits']

    def _sampling_distribution_logits(
        self,
        logits: torch.Tensor,
        top_k: Optional[int],
        top_p: float,
    ) -> torch.Tensor:
        if top_k is not None:
            return _top_k_logits(logits, k=top_k)
        if top_p < 1.0:
            return _top_p_logits(logits, p=top_p)
        return logits

    def teacher_forcing(
        self,
        source_texts: List[str],
        sample_ids: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        state, past_key_values = self._get_generation_cache(source_texts)

        sample_logits = []
        sample_log_probs = []
        sample_values = []
        for i in range(sample_ids.shape[-1]):
            policy_out = self._policy_forward(state)
            logits = policy_out['logits'] + self.logit_bias
            sampling_logits = self._sampling_distribution_logits(
                logits, top_k=top_k, top_p=top_p)

            actions = sample_ids[:, i]
            dist = torch.distributions.Categorical(logits=sampling_logits)
            sample_log_probs.append(dist.log_prob(actions).unsqueeze(1))
            sample_values.append(policy_out['value'].unsqueeze(1))
            sample_logits.append(logits.unsqueeze(dim=1))

            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]
            state, past_key_values = self._get_generation_cache(
                token_strs, past_key_values)

        return dict(
            sample_logits=torch.cat(sample_logits, dim=1),
            sample_log_probs=torch.cat(sample_log_probs, dim=1),
            sample_values=torch.cat(sample_values, dim=1),
            sample_ids=sample_ids,
        )

    def sample(
        self,
        source_texts: List[str],
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        sample_log_probs, sample_values = [], []
        for i in range(max_new_tokens):
            policy_out = self._policy_forward(state)
            logits = policy_out['logits'] + self.logit_bias
            sampling_logits = self._sampling_distribution_logits(
                logits, top_k=top_k, top_p=top_p)

            dist = torch.distributions.Categorical(logits=sampling_logits)
            actions = dist.sample()
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens):
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            sample_log_probs.append(dist.log_prob(actions).unsqueeze(1))
            sample_values.append(policy_out['value'].unsqueeze(1))

            state, past_key_values = self._get_generation_cache(
                token_strs, past_key_values)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_log_probs = torch.cat(sample_log_probs, dim=1)
        sample_values = torch.cat(sample_values, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        return dict(sample_tokens=sample_tokens,
                    sample_logits=sample_logits,
                    sample_log_probs=sample_log_probs,
                    sample_values=sample_values,
                    sample_ids=sample_ids,
                    sample_lengths=sample_lengths)

    def greedy_search(self,
                      source_texts: List[str],
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            policy_out = self._policy_forward(state)
            logits = policy_out['logits'] + self.logit_bias

            actions = logits.argmax(dim=-1)
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens):
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))

            state, past_key_values = self._get_generation_cache(
                token_strs, past_key_values)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        return dict(sample_tokens=sample_tokens,
                    sample_logits=sample_logits,
                    sample_ids=sample_ids,
                    sample_lengths=sample_lengths)

    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None):
        token_encoding = (self.generator
                          .tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt')
                          .to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = self.generator.model.transformer(input_ids,
                                                   past_key_values=past_key_values,
                                                   use_cache=True)
        last_token_hidden_state = \
            outputs.last_hidden_state[np.arange(input_ids.shape[0]),
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        return last_token_hidden_state, past_key_values

    def generate(
        self,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        assert num_beams == 1, "Beam search not supported yet"
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) and (num_beams == 1)
        is_sample_gen_mode = (do_sample == True) and (num_beams == 1)
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            return self.greedy_search(source_texts=source_texts,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id)
        elif is_sample_gen_mode:
            return self.sample(source_texts=source_texts,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id)
