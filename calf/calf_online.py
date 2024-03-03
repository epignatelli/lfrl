from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

import pickle
import re
import requests
from flax import struct
from jax import Array
import jax
from jax.random import KeyArray
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import io_callback

import flax.linen as nn
import optax
from helx.base.mdp import Timestep
from helx.base.spaces import Discrete
from helx.envs.environment import Environment

from .prompts import PROMPT_REDISTRIBUTION
from .ppo import PPO, HParams as PPOHparams, run_n_steps
from .buffer import RingBuffer
from .annotate import _compose_prompt, query_llm, _parse_redistribution,


class HParams(PPOHparams):
    online: bool = False
    buffer_capacity: int = 1_000
    """The maximum number of transitions to store in the replay buffer."""
    buffer_path: str = "./demonstrations.pkl"
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
        assert isinstance(env.action_space, Discrete)
        assert (
            env.observation_space.shape[0]
            == env.action_space.shape[0]
            == hparams.n_actors
        )
        actor = nn.Sequential(
            [
                encoder.clone(),
                nn.Dense(
                    env.action_space.maximum + 1,
                    kernel_init=nn.initializers.orthogonal(scale=0.01),
                ),
            ]
        )
        critic = nn.Sequential([encoder.clone(), nn.Dense(1)])
        unbatched_obs_sample = env.observation_space.sample(key)[0]
        params_actor = actor.init(key, unbatched_obs_sample)
        params_critic = critic.init(key, unbatched_obs_sample)
        params = {"actor": params_actor, "critic": params_critic}
        optimiser = optax.chain(
            optax.clip_by_global_norm(hparams.gradient_clip_norm),
            optax.adam(hparams.learning_rate),
        )

        train_state = {
            "opt_state": optimiser.init(params),
            "iteration": jnp.asarray(0),
            "params": params,
        }
        return cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            train_state=train_state,
        )

    def init_buffer(self, env: Environment, *, key: KeyArray) -> RingBuffer:
        # initialise buffer
        if self.hparams.online:
            timestep = env.reset(key)
            episodes, timestep = run_n_steps(
                env, timestep, self, self.hparams.transition_len, key=key
            )
        else:
            with open(self.hparams.buffer_path, "rb") as f:
                episodes = pickle.load(f)
                episodes = jtu.tree_map(lambda x: jnp.stack(x, axis=0), episodes)

        episodes = self.evaluate_experience(episodes[0])
        buffer = RingBuffer.create(
            episodes, self.hparams.buffer_capacity, self.hparams.transition_len
        )
        return buffer

    def calf_rewards(self, experience: Timestep) -> Timestep:
        """Replaces the rewards in `experience` with:
        `beta * rewards + (1 - beta) * episodes.reward`
        """
        # filter rewarding episodes
        batch_idx, time_idx = jnp.nonzero(experience.is_last())
        episodes = []  # list of only rewarding episodes
        for i in range(len(batch_idx)):
            episode = experience[
                batch_idx[i], time_idx[i] - self.hparams.transition_len : time_idx[i]
            ]
            if episode.reward.sum() > 1:
                episodes.append(episode)

        episodes = jtu.tree_map(lambda x: jnp.stack(x, axis=0), episodes)

        # query LLM
        rewards = query_llm(episodes, self.hparams.max_new_tokens)

        # replace rewards
        beta = self.hparams.llm_reward_coefficient
        new_rewards = beta * rewards + (1 - beta) * episodes.reward
        episodes = episodes.replace(reward=new_rewards)
        return episodes

    @jax.jit
    def update(
        self, experience: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        k1, k2 = jax.random.split(key)

        # PPO update
        self, log_ppo = super().update(experience, key=k1)

        # SIL buffer update
        buffer = self.train_state["buffer"]
        #  we update only if we learn online, otherwise
        if self.hparams.online:
            rewarding_episodes = io_callback(self.calf_rewards, experience, experience)
            # TODO: buffer.add_range is not jit-safe, pull it into the callback
            buffer = buffer.add_range(rewarding_episodes, len(rewarding_episodes))
            self.train_state["buffer"] = buffer

        # LLM updates
        draw = jax.random.uniform(k2, ())
        iteration = self.train_state["iteration"]
        llm_update_prob = optax.linear_schedule(1, 0, 5_000_000, 0)(iteration)
        cond = (
            len(buffer) * self.hparams.transition_len >= self.hparams.batch_size
            and draw < llm_update_prob
        )
        self, log_llm = jax.lax.cond(
            cond,
            super().update,
            lambda buffer, key: (self, log),
            (buffer, key),
        )

        # takes `log_llm[key]` if `key` exists in both
        log = {**log_ppo, **log_llm}

        return self, log
