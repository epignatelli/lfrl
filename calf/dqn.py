# # Copyright 2023 The Helx Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# from __future__ import annotations
# from functools import partial
# from typing import Dict, Tuple

# import distrax
# import jax
# import jax.numpy as jnp
# import jax.tree_util as jtu
# import optax
# import rlax
# from flax import linen as nn
# from flax import struct
# from flax.core.scope import VariableDict as Params
# from jax import Array
# from jax.random import KeyArray
# from jax.lax import stop_gradient
# from optax import GradientTransformation

# from helx.base.mdp import StepType, Timestep
# from helx.base.memory import ReplayBuffer
# from helx.base.spaces import Discrete
# from helx.envs.environment import Environment

# from .trial import Agent, HParams as HParamsBase, run_n_steps


# class HParams(HParamsBase):
#     # network
#     hidden_size: int = 128
#     # rl
#     initial_exploration: float = 1.0
#     final_exploration: float = 0.01
#     final_exploration_frame: int = 1000000
#     replay_start: int = 1000
#     replay_memory_size: int = 1000
#     update_frequency: int = 1
#     target_network_update_frequency: int = 10000
#     discount: float = 0.99
#     n_steps: int = 1
#     # sgd
#     batch_size: int = 32
#     learning_rate: float = 0.00025
#     gradient_momentum: float = 0.95
#     squared_gradient_momentum: float = 0.95
#     min_squared_gradient: float = 0.01
#     gradient_clip_norm: float = 0.5
#     """The maximum norm of the gradient"""


# class DQN(Agent):
#     """Implements a Deep Q-Network:
#     Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
#     Nature 518.7540 (2015): 529-533.
#     https://www.nature.com/articles/nature14236"""

#     hparams: HParams = struct.field(pytree_node=False)
#     optimiser: GradientTransformation = struct.field(pytree_node=False)
#     critic: nn.Module = struct.field(pytree_node=False)

#     @classmethod
#     def init(
#         cls,
#         env: Environment,
#         hparams: HParams,
#         encoder: nn.Module,
#         *,
#         key: Array,
#     ) -> DQN:
#         critic = nn.Sequential(
#             [
#                 encoder,
#                 nn.Dense(env.action_space.maximum + 1),
#             ]
#         )
#         unbatched_obs_sample = env.observation_space.sample(key)[0]
#         params = critic.init(key, unbatched_obs_sample)
#         params_target = jtu.tree_map(lambda x: x, params)  # copy params
#         optimiser = optax.chain(
#             optax.clip_by_global_norm(hparams.gradient_clip_norm),
#             optax.adam(hparams.learning_rate),
#         )
#         buffer = ReplayBuffer.create(
#             timestep,
#             hparams.replay_memory_size,
#             hparams.n_steps,
#         )
#         train_state = {
#             "opt_state": optimiser.init(params),
#             "iteration": jnp.asarray(0),
#             "params": params,
#             "params_target": params_target,
#             "buffer": buffer,
#         }
#         return DQN(
#             hparams=hparams, optimiser=optimiser, critic=critic, train_state=train_state
#         )

#     def policy(self, params: Params, observation: Array) -> distrax.EpsilonGreedy:
#         q_values = self.value_fn(params, observation)
#         eps = optax.polynomial_schedule(
#             init_value=self.hparams.initial_exploration,
#             end_value=self.hparams.final_exploration,
#             transition_steps=self.hparams.final_exploration_frame
#             - self.hparams.replay_start,
#             transition_begin=self.hparams.replay_start,
#             power=1.0,
#         )(self.train_state["iteration"])
#         return distrax.EpsilonGreedy(
#             preferences=q_values, epsilon=eps, dtype=jnp.int32  # type: ignore
#         )

#     def value_fn(self, params: Params, observation: Array) -> Array:
#         return jnp.asarray(self.critic.apply(params["critic"], observation))[0]

#     def collect_experience(
#         self, env: Environment, timestep: Timestep, *, key: KeyArray
#     ) -> Tuple[Timestep, Timestep]:
#         """Collects `n_actors` trajectories of experience of length `n_steps`
#         from the environment. This method is the only one that interacts with the
#         environment, and cannot be jitted unless the environment is JAX-compatible.
#         Args:
#             env (Environment): the environment must be a batched environment,
#                 such that `env.step` returns `n_actors` tim
#             esteps
#             key (Array): a random key to sample actions
#         Returns:
#             Timestep: a Timestep representing a batch of trajectories.
#             The first axis is the number of actors, the second axis is time.
#         """
#         return run_n_steps(env, timestep, self, self.hparams.n_steps, key=key)

#     def sample_experience(self, episodes: Timestep, *, key: KeyArray) -> Timestep:
#         """Samples a minibatch of transitions from the collected experience with
#         the "Shuffle transitions (recompute advantages)" method: see
#         https://arxiv.org/pdf/2006.05990.pdf
#         Args:
#             episodes (Timestep): a Timestep representing a batch of trajectories.
#                 The first axis is the number of actors, the second axis is time.
#             key (Array): a random key to sample actions
#         Returns:
#             Timestep: a minibatch of transitions (2-steps sars tuples) where the
#             first axis is the number of actors * n_steps // n_minibatches
#         """
#         assert episodes.t.ndim == 2, "episodes.ndim must be 2, got {} instead.".format(
#             episodes.t.ndim
#         )
#         batch_size = self.hparams.batch_size
#         # sample transitions
#         batch_len = episodes.t.shape[0]
#         actor_idx = jax.random.randint(
#             key, shape=(batch_size,), minval=0, maxval=batch_len
#         )
#         # exclude last timestep (-1) as it was excluded from the advantage computation
#         episode_length = episodes.t.shape[1]
#         time_idx = jax.random.randint(
#             key, shape=(batch_size,), minval=0, maxval=episode_length
#         )
#         # using `at_time`` rather than standard indexing returns the timestep
#         # aligned in time, e.g., (s_t, a_t, r_t), rather than (s_t, a_{t-1}, r_{t-1})
#         return episodes.at_time[actor_idx, time_idx]  # (batch_size,)

#     def evaluate_experience(self, episodes: Timestep) -> Timestep:
#         """Calculate targets for multiple concatenated episodes.
#         For episodes with a total sum of T steps, it computes values and advantage
#         in the range [0, T-1]"""
#         assert (
#             episodes.t.ndim == 1
#         ), "episodes.t.ndim must be 1, got {} instead.".format(episodes.t.ndim)
#         # s_t \\forall t \\in [0, T]
#         obs = episodes.observation
#         # r_t \\forall t \\in [0, T-1]
#         rewards = episodes.reward
#         # \\gamma^t \\forall t \\in [0, T-1]
#         discounts = episodes.is_mid() * self.hparams.discount**episodes.t
#         # q(s_t, a'_t) \\forall a'
#         values = jax.vmap(self.value_fn, in_axes=(None, 0))(
#             self.train_state["params_target"], obs
#         ) * (episodes.step_type != StepType.TERMINATION)
#         # return target from [0, T-1]
#         returns = rlax.discounted_returns(rewards, discounts, values)
#         episodes.info["return_target"] = returns
#         return episodes

#     def loss(
#         self, params: Params, transition: Timestep
#     ) -> Tuple[Array, Dict[str, Array]]:
#         assert "return_target" in transition.info
#         observation = transition.observation
#         return_target = stop_gradient(transition.info["return_target"])
#         action = transition.action

#         value = self.value_fn(params, observation)

#         q_tm1 = jnp.asarray(self.critic.apply(params, s_tm1))
#         q_t = jnp.asarray(self.critic.apply(params_target, s_t)) * terminal_tm1

#         rlax.n_step_bootstrapped_returns()
#         td_error = rlax.q_learning(
#             q_tm1, a_tm1, r_t, discount_t, q_t, stop_target_gradients=True
#         )
#         td_error = jnp.asarray(td_error)
#         td_loss = jnp.mean(0.5 * td_error**2)
#         return td_loss

#     @partial(jax.jit, static_argnums=2)
#     def update(
#         self, episodes: Timestep, *, key: KeyArray
#     ) -> Tuple[DQN, Dict[str, Array]]:
#         params, train_state, log = self.params, self.train_state, {}
#         params_target = train_state["params_target"]
#         iteration = train_state["iteration"]
#         buffer = train_state["buffer"]
#         opt_state = train_state["opt_state"]

#         def batch_loss(params, transitions):
#             out = jax.vmap(self.loss, in_axes=(None, 0))(params, transitions)
#             out = jtu.tree_map(lambda x: x.mean(axis=0), out)
#             return out

#         def _sgd_step(params):
#             transitions = buffer.sample(key, self.hparams.batch_size)
#             loss_fn = lambda params, trans: jnp.mean(
#                 jax.vmap(batch_loss, in_axes=(None, 0, None))(
#                     params, trans
#                 )
#             )
#             loss, grads = jax.value_and_grad(loss_fn)(params, transitions)
#             updates, opt_state = self.optimiser.update(grads, opt_state)
#             params = optax.apply_updates(params, updates)
#             return params, opt_state, loss

#         # update buffer
#         buffer = buffer.add(episodes)

#         cond = buffer.size() < self.hparams.replay_memory_size
#         cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
#         params, opt_state, loss = jax.lax.cond(
#             cond,
#             lambda p, o: _sgd_step(p),
#             lambda p, o: (p, o, jnp.asarray(float("inf"))),
#             train_state.params,
#             train_state.opt_state,
#         )
#         # sample batch of transitions from the buffer
#         transitions = self.sample_experience(buffer, key=key)
#         # calculate bootstrapped n_step returns
#         transitions = jax.vmap(self.evaluate_experience)(transitions)
#         # SGD with DQN loss
#         (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
#             params, transitions
#         )
#         updates, opt_state = self.optimiser.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         # Update target network params
#         params_target = optax.periodic_update(
#             params,
#             params_target,
#             iteration,
#             self.hparams.target_network_update_frequency,
#         )
#         # update train_state object
#         train_state["iteration"] += 1
#         train_state["opt_state"] = opt_state
#         train_state["params"] = params
#         train_state["params_target"] = params_target

#         # log gradients norm
#         log["losses/grad_norm"] = optax.global_norm(updates)
#         if "buffer" in train_state:
#             log["buffer_size"] = len(train_state["buffer"])
#         return self.replace(train_state=train_state), log

#     # def update(
#     #     self,
#     #     train_state: DQNState,
#     #     transition: Timestep,
#     #     *,
#     #     key: KeyArray,
#     # ) -> DQNState:
#     #     # update iteration
#     #     iteration = jnp.asarray(train_state.iteration + 1, dtype=jnp.int32)

#     #     # update memory
#     #     buffer = train_state.buffer.add(transition)

#     #     # update critic
#     #     def _sgd_step(params, opt_state):
#     #         transitions = buffer.sample(key, self.hparams.batch_size)
#     #         loss_fn = lambda params, trans: jnp.mean(
#     #             jax.vmap(self.loss, in_axes=(None, 0, None))(
#     #                 params, trans, train_state.params_target
#     #             )
#     #         )
#     #         loss, grads = jax.value_and_grad(loss_fn)(params, transitions)
#     #         updates, opt_state = self.optimiser.update(grads, opt_state)
#     #         params = optax.apply_updates(params, updates)
#     #         return params, opt_state, loss

#     #     cond = buffer.size() < self.hparams.replay_memory_size
#     #     cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
#     #     params, opt_state, loss = jax.lax.cond(
#     #         cond,
#     #         lambda p, o: _sgd_step(p, o),
#     #         lambda p, o: (p, o, jnp.asarray(float("inf"))),
#     #         train_state.params,
#     #         train_state.opt_state,
#     #     )

#     #     # update target critic
#     #     params_target = optax.periodic_update(
#     #         params,
#     #         train_state.params_target,
#     #         jnp.asarray(iteration, dtype=jnp.int32),
#     #         self.hparams.target_network_update_frequency,
#     #     )

#     #     # log
#     #     log = DQNLog(
#     #         iteration=iteration,
#     #         critic_loss=loss,
#     #         step_type=transition.step_type[-1],
#     #         returns=train_state.log.returns
#     #         + jnp.sum(
#     #             self.hparams.discount ** transition.t[:-1] * transition.reward[:-1]
#     #         ),
#     #         buffer_size=buffer.size(),
#     #     )

#     #     # update train_state
#     #     train_state = train_state.replace(
#     #         iteration=iteration,
#     #         opt_state=opt_state,
#     #         params=params,
#     #         params_target=params_target,
#     #         buffer=buffer,
#     #         log=log,
#     #     )
#     #     return train_state
