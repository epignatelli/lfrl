from __future__ import annotations
from typing import Dict, Tuple

from flax import struct
from jax import Array
import jax
from jax.random import KeyArray

import flax.linen as nn
from helx.base.mdp import Timestep
from helx.envs.environment import Environment

from .ppo import PPO, HParams as PPOHparams
from .io import load_pickle_stream


class HParams(PPOHparams):
    """The maximum number of transitions to store in the replay buffer."""

    buffer_path: str = "/scratch/uceeepi/calf/redistribution_alt/demonstrations_1.pkl2"
    llm_reward_coefficient: int = 1
    llm_reward_epochs: int = 20


class CALF(PPO):
    hparams: HParams = struct.field(pytree_node=False)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: KeyArray,
    ) -> CALF:
        # initialise PPO
        ppo_agent = super().init(env, hparams, encoder, key=key)
        # load buffer
        train_state = ppo_agent.train_state
        train_state["buffer"] = load_pickle_stream(hparams.buffer_path)
        # create object
        self = cls(
            hparams=hparams,
            optimiser=ppo_agent.optimiser,
            actor=ppo_agent.actor,
            critic=ppo_agent.critic,
            train_state=train_state,
        )
        return self

    def update(
        self, experience: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        k1, k2 = jax.random.split(key)

        # PPO update
        batch_idx = jax.random.randint(
            key, shape=(self.hparams.batch_size,), minval=0, maxval=len(experience.t)
        )
        experience = experience[batch_idx]
        self, log_ppo = super().update(experience, key=k1)

        # CALF updates
        buffer = self.train_state["buffer"]
        batch_idx = jax.random.randint(
            key, shape=(self.hparams.batch_size,), minval=0, maxval=len(buffer.t)
        )
        buffer = buffer[batch_idx]
        self, log_llm = super().update(
            buffer, n_epochs=self.hparams.llm_reward_epochs, key=k2
        )

        # takes `log_llm[key]` if `key` exists in both
        log = {**log_ppo, **log_llm}

        return self, log
