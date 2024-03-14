from __future__ import annotations

from typing import Any, Dict, Tuple
from functools import partial
import time

import numpy as np
import wandb

import jax
from jax.random import KeyArray
from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu

import flax.linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
import distrax
from helx.envs.environment import Environment, Timestep


class HParams(struct.PyTreeNode):
    discount: float
    iteration_size: int


class Agent(struct.PyTreeNode):
    train_state: Dict[str, Any]
    hparams: HParams

    @property
    def params(self) -> Params:
        assert "params" in self.train_state
        return self.train_state["params"]  # type: ignore

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: KeyArray,
    ) -> Agent: ...

    def policy(self, params: Params, observation: Array) -> distrax.Distribution: ...

    def collect_experience(
        self, env: Environment, timestep: Timestep, *, key: KeyArray
    ) -> Tuple[Timestep, Timestep]: ...

    def update(
        self, trajectories: Timestep, *, key: KeyArray
    ) -> Tuple[Agent, Dict[str, Array]]: ...


class Experiment:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        wandb.init(project=name, config=config)

    def collect_experience(
        self, agent: Agent, env: Environment, timestep: Timestep, *, key: KeyArray
    ) -> Tuple[Timestep, Timestep]:
        experience, timestep = agent.collect_experience(env, timestep, key=key)
        return experience, timestep

    def update(
        self, agent: Agent, experience: Timestep, *, key: KeyArray
    ) -> Tuple[Agent, Dict[str, Array]]:
        agent, log = agent.update(experience, key=key)
        return agent, log

    def close(self): ...

    def run(
        self,
        agent: Agent,
        env: Environment,
        key: KeyArray,
        render_log_rate: int = 1_000_000,
    ) -> Agent:
        budget = self.config["budget"]

        frames = jnp.asarray(0)
        iteration = jnp.asarray(0)
        timestep = env.reset(key)
        if "return" not in timestep.info:
            timestep.info["return"] = timestep.reward
        while frames < budget:
            # step
            start_time = time.time()
            k1, k2, key = jax.random.split(key, num=3)
            experience, timestep = self.collect_experience(agent, env, timestep, key=k1)
            agent, log = self.update(agent, experience, key=k2)

            # log frames and iterations
            log["frames"] = frames
            log["iteration"] = iteration
            log["fps"] = jnp.asarray(experience.t.size / (time.time() - start_time))

            # log episode length
            final_t = experience.t[experience.is_last()]
            if final_t.size > 0:
                log["train/episode_length"] = jnp.mean(final_t)
                log["train/min_episode_length"] = jnp.min(final_t)
                log["train/max_episode_length"] = jnp.max(final_t)

            # log rewards
            log["train/average_reward"] = jnp.mean(experience.reward)
            log["train/min_reward"] = jnp.min(experience.reward)
            log["train/max_reward"] = jnp.max(experience.reward)

            # log returns
            if "return" in experience.info:
                return_ = experience.info["return"]
                return_ = return_[experience.is_last()]
                if return_.size > 0:
                    log["train/return"] = jnp.mean(return_)
                    log["train/min_return"] = jnp.min(return_)
                    log["train/max_return"] = jnp.max(return_)

            # log success rates
            success_hits = jnp.sum(experience.reward == 1.0, dtype=jnp.int32)
            log["train/success_hits"] = success_hits
            log["train/success_rate"] = success_hits / jnp.sum(experience.is_last())

            # log render
            if frames % render_log_rate <= (agent.hparams.iteration_size + 1):
                start_t = experience.t[0][experience.is_first()[0]]
                end_t = experience.t[0][experience.is_last()[0]]
                if start_t.size > 0 and end_t.size > 0:
                    render = experience.observation[0, start_t[0] : end_t[0]]
                    log["train/render"] = wandb.Video(  #  type: ignore
                        np.asarray(render.transpose(0, 3, 1, 2)), fps=1
                    )

            # print and push log
            print(log)
            wandb.log(log)

            frames += experience.t.size
            iteration += 1

        self.close()
        return agent


def run_n_steps(
    env: Environment, timestep: Timestep, agent: Agent, n_steps: int, *, key: KeyArray
) -> Tuple[Timestep, Timestep]:
    """Runs `n_steps` in the environment using the agent's policy and returns a
    partial trajectory.

    Args:
        env (Environment): the environment to step in.
        timestep (Timestep): the timestep to start from.
        agent (PPO): the agent.
        n_steps (int): the number of steps to run.
        key (KeyArray): a random key to sample actions.

    Returns:
        Tuple[Timestep, Timestep]: a partial trajectory of length `n_steps` and the
        final timestep.
        The trajectory can contain multiple episodes.
        Each timestep in the trajectory has the following structure:
        $(t, o_t, a_{t-1}, r_{t-1}, step_type_{t-1}, info_{t-1})$
        where:
        $a_{t} \\sim \\pi(s_t)$ is the action sampled from the policy conditioned on
        $s_t$; $r_{t} = R(s_t, a_t)$ is the reward after taking the action $a_t$ in
        $s_t$.

    Example:
    0    (0, s_0,  -1, 0.0,      0, info_0, G_0)  RESET
    1    (1, s_1, a_0, r_0, term_0, info_0, G_1 = G_0 + r_0 * gamma ** 1)
    2    (2, s_2, a_1, r_1, term_1, info_1, G_2 = G_1 + r_1)
    3    (3, s_3, a_2, r_2, term_2, info_2, G_3 = G_2 + r_2)  TERMINATION
    4    (0, s_0,  -1, 0.0,      0, info_3, G_4 = G_3 + 0.0)  RESET
    5    (1, s_1, a_0, r_0, term_0, info_0, G_0 = r_0)
    6    (2, s_2, a_1, r_1, term_1, info_1, G_1 = G_1 + r_1)  TERMINATION
    7    (0, s_0,  -1, 0.0,      0, info_2, G_2 = G_1 + 0.0)  RESET

    idx = 3:
        timestep_t =   (3, s_3, a_2, r_2, term_2, info_2, G_2 = G_1 + r_2)  TERMINATION
        timestep_tp1 = (0, s_0,  -1, 0.0,      0, info_3, G_3 = G_2 + 0.0)  RESET
    """

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def policy_sample(params, observation, key):
        action_distribution = agent.policy(params, observation)
        action, log_prob = action_distribution.sample_and_log_prob(
            seed=key, sample_shape=env.action_space.shape[1:]
        )
        return action, log_prob

    episode = []
    for _ in range(n_steps):
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episode.append(timestep)
        # step the environment
        action, log_prob = policy_sample(agent.params, timestep.observation, k1)
        timestep.info["log_prob"] = log_prob
        next_timestep = env.step(k2, timestep, action)

        # log return, if available
        reward = next_timestep.reward
        # depure return log from intrinsic rewards
        if "intrinsic_reward" in timestep.info:
            reward = reward - timestep.info["intrinsic_reward"]
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
            ) + (reward * agent.hparams.discount**timestep.t)

        timestep = next_timestep

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode), timestep


def run_episode(env: Environment, agent: Agent, *, key: KeyArray) -> Timestep:
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def policy_sample(params, observation, key):
        action_distribution = agent.policy(params, observation)
        action, log_prob = action_distribution.sample_and_log_prob(
            seed=key, sample_shape=env.action_space.shape[1:]
        )
        return action, log_prob

    timestep = env.reset(key)
    timestep.info["return"] = timestep.reward
    final = timestep.is_last()
    episode = []
    while True:
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episode.append(timestep)
        # step the environment
        action, log_prob = policy_sample(agent.params, timestep.observation, k1)
        timestep.info["log_prob"] = log_prob
        next_timestep = env.step(k2, timestep, action)
        # log return, if available
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
                + (next_timestep.reward * agent.hparams.discount**timestep.t)
            )
        final = jnp.logical_or(final, timestep.is_last())
        if final.all():
            break
        timestep = next_timestep

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)
