from __future__ import annotations

from typing import Any, Dict, Tuple, Union
from functools import partial
import time

import numpy as np
import optax
import wandb

import jax
from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu

import flax.linen as nn
from flax import struct
from flax.typing import VariableDict as Params
import distrax
from helx.envs.environment import Environment, Timestep

RNNState = Dict[str, Tuple[Array, Array]]


class HParams(struct.PyTreeNode):
    discount: float


class AgentState(struct.PyTreeNode):
    iteration: Array
    params: Params
    opt_state: optax.OptState
    hidden_state: RNNState = struct.field(default_factory=dict)
    buffer: Timestep | None = None


class Agent(struct.PyTreeNode):
    hparams: HParams

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: Array,
    ) -> Agent: ...

    def policy(
        self, params: Params, observation: Array, *, hidden_state: RNNState
    ) -> Tuple[RNNState, distrax.Distribution]: ...

    def collect_experience(
        self,
        env: Environment,
        timestep: Timestep,
        params: Params,
        *,
        key: Array,
        hidden_state: RNNState,
    ) -> Tuple[Timestep, Timestep, RNNState]: ...

    def update(
        self,
        agent_state: AgentState,
        episodes: Timestep,
        n_epochs: int | None = None,
        *,
        key: Array,
    ) -> Tuple[AgentState, Dict[str, Array]]: ...


class Experiment:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        wandb.init(project=name, config=config)

    def collect_experience(
        self,
        agent: Agent,
        env: Environment,
        timestep: Timestep,
        agent_state: AgentState,
        *,
        key: Array,
        hidden_state: RNNState,
    ) -> Tuple[Timestep, Timestep, RNNState]:
        experience, timestep, hidden_state = agent.collect_experience(
            env, timestep, agent_state.params, key=key, hidden_state=hidden_state
        )
        return experience, timestep, hidden_state

    def update(
        self, agent: Agent, agent_state: AgentState, experience: Timestep, *, key: Array
    ) -> Tuple[AgentState, Dict[str, Array]]:
        agent_state, log = agent.update(agent_state, experience, key=key)
        return agent_state, log

    def close(self):
        wandb.finish()

    def run(
        self,
        agent: Tuple[Agent, AgentState],
        env: Environment,
        key: Array,
        render_log_rate: int = 1_000_000,
    ) -> AgentState:
        budget = self.config["budget"]
        agent_, agent_state = agent
        frames = jnp.asarray(0)
        iteration = jnp.asarray(0)
        timestep = env.reset(key)
        hidden_state = agent_state.hidden_state
        if "return" not in timestep.info:
            timestep.info["return"] = timestep.reward
        while frames < budget:
            # step
            start_time = time.time()
            k1, k2, key = jax.random.split(key, num=3)
            experience, timestep, hidden_state = self.collect_experience(
                agent_, env, timestep, agent_state, key=k1, hidden_state=hidden_state
            )
            agent_state, log = self.update(agent_, agent_state, experience, key=k2)

            # log frames and iterations
            log["frames"] = frames
            log["iteration"] = iteration
            log["fps"] = jnp.asarray(experience.t.size / (time.time() - start_time))

            # log episode length
            final_t = experience.t[experience.is_last()]
            if final_t.size > 0:
                log["episode_length/avg_episode_length"] = jnp.mean(final_t)
                log["episode_length/min_episode_length"] = jnp.min(final_t)
                log["episode_length/max_episode_length"] = jnp.max(final_t)

            # log rewards
            log["reward/avg_reward"] = jnp.mean(experience.reward)
            log["reward/min_reward"] = jnp.min(experience.reward)
            log["reward/max_reward"] = jnp.max(experience.reward)

            if "intrinsic_reward" in experience.info:
                int_reward = experience.info["intrinsic_reward"]
                log["reward/avg_intrinsic_reward"] = jnp.mean(int_reward)
                log["reward/min_intrinsic_reward"] = jnp.min(int_reward)
                log["reward/max_intrinsic_reward"] = jnp.max(int_reward)
                ext_reward = experience.reward - int_reward
                log["reward/avg_extrinsic_reward"] = jnp.mean(ext_reward)
                log["reward/min_extrinsic_reward"] = jnp.min(ext_reward)
                log["reward/max_extrinsic_reward"] = jnp.max(ext_reward)

            # log returns
            if "return" in experience.info:
                return_ = experience.info["return"]
                return_ = return_[experience.is_last()]
                if return_.size > 0:
                    log["return/avg_return"] = jnp.mean(return_)
                    log["return/min_return"] = jnp.min(return_)
                    log["return/max_return"] = jnp.max(return_)

                    regret = env.reward_space.maximum - return_
                    log["return/avg_regret"] = jnp.mean(regret)
                    log["return/min_regret"] = jnp.min(regret)
                    log["return/max_regret"] = jnp.max(regret)

            # log success rates
            success_hits = jnp.sum(experience.reward == 1.0, dtype=jnp.int32)
            log["success/success_hits"] = success_hits
            log["success/success_rate"] = success_hits / jnp.sum(experience.is_last())

            # log render
            if (frames % render_log_rate) and (
                experience.observation[0, 0].ndim == 3
                and experience.observation[0, 0].shape[-1] == 3
            ):
                start_t = experience.t[0][experience.is_first()[0]]
                end_t = experience.t[0][experience.is_last()[0]]
                if start_t.size > 0 and end_t.size > 0:
                    render = experience.observation[0, start_t[0] : end_t[0]]
                    log["vis/render"] = wandb.Video(  #  type: ignore
                        np.asarray(render.transpose(0, 3, 1, 2)), fps=1
                    )

            # print and push log
            print(log)
            wandb.log(log)

            frames += experience.t.size
            iteration += 1

        self.close()
        return agent_state


# def run_n_steps(
#     env: Environment, timestep: Timestep, agent: Agent, n_steps: int, *, key: Array
# ) -> Tuple[Timestep, Timestep]:
#     """Runs `n_steps` in the environment using the agent's policy and returns a
#     partial trajectory.

#     Args:
#         env (Environment): the environment to step in.
#         timestep (Timestep): the timestep to start from.
#         agent (PPO): the agent.
#         n_steps (int): the number of steps to run.
#         key (Array): a random key to sample actions.

#     Returns:
#         Tuple[Timestep, Timestep]: a partial trajectory of length `n_steps` and the
#         final timestep.
#         The trajectory can contain multiple episodes.
#         Each timestep in the trajectory has the following structure:
#         $(t, o_t, a_{t-1}, r_{t-1}, step_type_{t-1}, info_{t-1})$
#         where:
#         $a_{t} \\sim \\pi(s_t)$ is the action sampled from the policy conditioned on
#         $s_t$; $r_{t} = R(s_t, a_t)$ is the reward after taking the action $a_t$ in
#         $s_t$.

#     Example:
#     0    (0, s_0,  -1, 0.0,      0, info_0, G_0)  RESET
#     1    (1, s_1, a_0, r_0, term_0, info_0, G_1 = G_0 + r_0 * gamma ** 1)
#     2    (2, s_2, a_1, r_1, term_1, info_1, G_2 = G_1 + r_1)
#     3    (3, s_3, a_2, r_2, term_2, info_2, G_3 = G_2 + r_2)  TERMINATION
#     4    (0, s_0,  -1, 0.0,      0, info_3, G_4 = G_3 + 0.0)  RESET
#     5    (1, s_1, a_0, r_0, term_0, info_0, G_0 = r_0)
#     6    (2, s_2, a_1, r_1, term_1, info_1, G_1 = G_1 + r_1)  TERMINATION
#     7    (0, s_0,  -1, 0.0,      0, info_2, G_2 = G_1 + 0.0)  RESET

#     idx = 3:
#         timestep_t =   (3, s_3, a_2, r_2, term_2, info_2, G_2 = G_1 + r_2)  TERMINATION
#         timestep_tp1 = (0, s_0,  -1, 0.0,      0, info_3, G_3 = G_2 + 0.0)  RESET
#     """

#     def policy_sample(
#         params: Params,
#         observation: Array,
#         shape: Tuple[int, ...],
#         key: Array,
#         hidden_state: RNNState,
#     ) -> Tuple[RNNState, Array, Array]:
#         hidden_state, action_distribution = agent.policy(
#             params, observation, hidden_state=hidden_state
#         )
#         action, log_prob = action_distribution.sample_and_log_prob(
#             seed=key, sample_shape=shape
#         )
#         return hidden_state, jnp.asarray(action), jnp.asarray(log_prob)

#     is_batched_env = env.ndim > 0
#     if is_batched_env:
#         action_shape = env.action_space.shape[1:]
#         policy_sample = jax.vmap(policy_sample, in_axes=(None, 0, None, 0, 0))

#     else:
#         action_shape = env.action_space.shape

#     policy_sample = jax.jit(policy_sample, static_argnames=["shape"])

#     hidden_state = timestep.info.get("hidden_state", None)

#     episode = []
#     for _ in range(n_steps):
#         k1, k2, key = jax.random.split(key, num=3)
#         if is_batched_env:
#             k1 = jax.random.split(k1, num=env.action_space.shape[0])
#         episode.append(timestep)
#         # step the environment
#         hidden_state, action, log_prob = policy_sample(
#             params=agent.params,
#             observation=timestep.observation,
#             shape=action_shape,
#             key=k1,
#             hidden_state=hidden_state,
#         )
#         # log action log_prob and hidden state
#         timestep.info["log_prob"] = log_prob
#         timestep.info["hidden_state"] = hidden_state
#         next_timestep = env.step(k2, timestep, action)

#         # depure return log from intrinsic rewards
#         reward = next_timestep.reward
#         if "intrinsic_reward" in timestep.info:
#             timestep.info["total_reward"] = reward
#             reward = reward - timestep.info["intrinsic_reward"]

#         # log returns
#         if "return" in timestep.info:
#             next_timestep.info["return"] = timestep.info["return"] * (
#                 timestep.is_mid()
#             ) + (reward * agent.hparams.discount**timestep.t)

#         timestep = next_timestep

#     batched = jtu.tree_map(lambda *x: jnp.stack(x, axis=env.ndim), *episode)
#     return batched, timestep
