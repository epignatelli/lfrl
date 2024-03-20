# Copyright 2023 The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
from jax import Array
from jax.random import KeyArray
from optax import GradientTransformation, OptState
from helx.base.mdp import StepType, Timestep
from helx.envs.environment import Environment

from .trial import Agent, HParams as HParamsBase, run_n_steps
from .buffer import RingBuffer


class HParams(HParamsBase):
    # rl
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    final_exploration_frame: int = 1000000
    replay_start: int = 1000
    replay_memory_size: int = 1000
    update_frequency: int = 1
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    n_steps: int = 1
    # sgd
    batch_size: int = 32
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    gradient_clip_norm: float = 0.5
    num_envs: int = 1
    """The maximum norm of the gradient"""


class DQN(Agent):
    """Implements a Deep Q-Network:
    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    https://www.nature.com/articles/nature14236"""

    hparams: HParams = struct.field(pytree_node=False)
    optimiser: GradientTransformation = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: Array,
    ) -> DQN:
        critic = nn.Sequential(
            [
                encoder,
                nn.Dense(env.action_space.maximum + 1),
            ]
        )
        print("Initialising networks...\t", end="", flush=True)
        obs_sample = env.observation_space.sample(key)
        if env.is_batched():
            obs_sample = obs_sample[0]
        params = critic.init(key, obs_sample)
        params_target = jtu.tree_map(lambda x: x, params)  # copy params
        optimiser = optax.adam(hparams.learning_rate)
        train_state = {
            "opt_state": optimiser.init(params),
            "iteration": jnp.asarray(0),
            "params": params,
            "params_target": params_target,
        }
        print("Done")

        # initialise buffer
        print("Initialising buffer...\t", end="", flush=True)
        agent = DQN(
            hparams=hparams, optimiser=optimiser, critic=critic, train_state=train_state
        )
        timestep = env.reset(key)
        item = run_n_steps(env, timestep, agent, hparams.n_steps + 1, key=key)[0]
        if env.is_batched():
            item = item[0]
        buffer = RingBuffer.init(item, hparams.replay_memory_size)
        print("Done")

        # inject buffer and return agent
        train_state["buffer"] = buffer

        # warmup ot fill buffer
        print("Warming up buffer...\t", end="", flush=True)
        n_iters = hparams.replay_memory_size // hparams.num_envs + 1
        for i in range(n_iters):
            print(f"Filling buffer {i}/{n_iters - 1}", end="\r")
            _, key = jax.random.split(key)  # type: ignore
            experience, timestep = agent.collect_experience(env, timestep, key=key)
            buffer = buffer.add(experience)
        print("\nDone")

        train_state["buffer"] = buffer
        return DQN(
            hparams=hparams, optimiser=optimiser, critic=critic, train_state=train_state
        )

    def policy(self, params: Params, observation: Array) -> distrax.EpsilonGreedy:
        q_values = self.value_fn(params, observation)
        eps = optax.polynomial_schedule(
            init_value=self.hparams.initial_exploration,
            end_value=self.hparams.final_exploration,
            transition_steps=(
                self.hparams.final_exploration_frame - self.hparams.replay_start
            ),
            transition_begin=self.hparams.replay_start,
            power=1.0,
        )(self.train_state["iteration"])
        return distrax.EpsilonGreedy(
            preferences=q_values, epsilon=eps, dtype=jnp.int32  # type: ignore
        )

    def value_fn(self, params: Params, observation: Array) -> Array:
        return jnp.asarray(self.critic.apply(params, observation))

    def collect_experience(
        self, env: Environment, timestep: Timestep, *, key: KeyArray
    ) -> Tuple[Timestep, Timestep]:
        """Collects `n_actors` trajectories of experience of length `n_steps`
        from the environment. This method is the only one that interacts with the
        environment, and cannot be jitted unless the environment is JAX-compatible.
        Args:
            env (Environment): the environment must be a batched environment,
                such that `env.step` returns `n_actors` tim
            esteps
            key (Array): a random key to sample actions
        Returns:
            Timestep: a Timestep representing a batch of trajectories.
            The first axis is the number of actors, the second axis is time.
        """
        return run_n_steps(env, timestep, self, self.hparams.n_steps + 1, key=key)

    def sample_experience(self, buffer: RingBuffer, *, key: KeyArray) -> Timestep:
        """Samples a minibatch of transitions from the collected experience with
        the "Shuffle transitions (recompute advantages)" method: see
        https://arxiv.org/pdf/2006.05990.pdf
        Args:
            experience (Timestep): a Timestep representing a batch of trajectories.
                The first axis is the number of actors, the second axis is time.
            key (Array): a random key to sample actions
        Returns:
            Timestep: a minibatch of transitions (2-steps sars tuples) where the
            first axis is the number of actors * n_steps // n_minibatches
        """
        batch_idx = jax.random.randint(
            key, shape=(self.hparams.batch_size,), minval=0, maxval=buffer.length()
        )
        return buffer[batch_idx]  # (B, 2)

    def loss(self, params: Params, transition: Timestep) -> Array:
        s_tm1 = transition.observation[0]
        r_tm1 = transition.reward[0]
        a_tm1 = transition.action[0]
        s_t = transition.observation[1]
        term_t = transition.step_type[1] == StepType.TERMINATION

        q_tm1 = self.value_fn(params, s_tm1)
        q_t = self.value_fn(self.train_state["params_target"], s_t)
        y_tm1 = r_tm1 + self.hparams.discount * jnp.max(q_t) * term_t
        y_tm1 = jax.lax.stop_gradient(y_tm1)
        td_loss = 0.5 * jnp.square(q_tm1[a_tm1] - y_tm1)
        return td_loss

    @jax.jit
    def update(
        self, experience: Timestep, *, key: Array
    ) -> Tuple[DQN, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0), axis_name="batch")(
                params, transitions
            )
            return jnp.mean(out)

        def _sgd_step(params, opt_state):
            transitions = self.sample_experience(buffer, key=key)
            loss, grads = jax.value_and_grad(batch_loss)(params, transitions)
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        train_state = self.train_state
        params: Params = train_state["params"]
        params_target = train_state["params_target"]
        iteration: Array = train_state["iteration"]
        buffer: RingBuffer = train_state["buffer"]
        opt_state: OptState = train_state["opt_state"]
        log = {}

        # update buffer
        buffer = buffer.add(experience)

        # update online network
        cond = buffer.length() < self.hparams.replay_memory_size
        cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
        params, opt_state, loss = jax.lax.cond(
            cond,
            lambda p, o: (p, o, jnp.asarray(float("inf"))),
            lambda p, o: _sgd_step(p, o),
            params,
            opt_state,
        )

        # Update target network
        params_target = optax.periodic_update(
            params,
            params_target,
            iteration,
            self.hparams.target_network_update_frequency,
        )

        # update train_state object
        train_state["params"] = params
        train_state["params_target"] = params_target
        train_state["iteration"] += 1
        train_state["buffer"] = buffer
        train_state["opt_state"] = opt_state

        # update log
        log["buffer_size"] = buffer.length()
        log["losses/critic"] = loss

        return self.replace(train_state=train_state), log
