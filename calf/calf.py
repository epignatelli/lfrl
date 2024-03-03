from __future__ import annotations
from typing import Dict, Tuple

import pickle
from flax import struct
from jax import Array
import jax
from jax.random import KeyArray
import jax.tree_util as jtu
import jax.numpy as jnp

import flax.linen as nn
from helx.base.mdp import Timestep
from helx.envs.environment import Environment

from .ppo import PPO, HParams as PPOHparams


class HParams(PPOHparams):
    online: bool = False
    buffer_capacity: int = 1_000
    """The maximum number of transitions to store in the replay buffer."""
    buffer_path: str = "./annotations/redistribution_alt.pkl"
    transition_len: int = 10
    max_new_tokens: int = 512
    llm_reward_coefficient: int = 1


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
        with open(hparams.buffer_path, "rb") as f:
            buffer = tuple(pickle.load(f))
        n_batches = len(buffer) // hparams.n_actors
        upper = n_batches * hparams.n_actors
        buffer = buffer[:upper]
        buffer = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *buffer)
        buffer = jtu.tree_map(lambda x: jnp.split(x, n_batches, axis=0), buffer)
        train_state["buffer"] = buffer
        # create object
        self = cls(
            hparams=hparams,
            optimiser=ppo_agent.optimiser,
            actor=ppo_agent.actor,
            critic=ppo_agent.critic,
            train_state=train_state,
        )
        return self

    @jax.jit
    def update(
        self, experience: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        k1, k2 = jax.random.split(key)

        # PPO update
        self, log_ppo = super().update(experience, key=k1)

        # LLM updates
        buffer = self.train_state["buffer"]
        self, log_llm = super().update(buffer, key=k2)

        # takes `log_llm[key]` if `key` exists in both
        log = {**log_ppo, **log_llm}

        return self, log
