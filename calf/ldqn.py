from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
import jax.tree_util as jtu
import optax
from flax import struct

from helx.base.spaces import Discrete
from helx.base.mdp import Timestep
from helx.agents import DQN, DQNState, DQNLog, DQNHParams

from .questions import ask_credit, parse_credit
from .buffer import RingBuffer


class LDQNHParams(DQNHParams):
    language_update_probability: float = 0.5
    llm_batch_size: int = 32
    n_llms: int = 4


class LDQNState(DQNState):
    annotation_buffer: RingBuffer
    buffer: RingBuffer


class Annotation(struct.PyTreeNode):
    timestep: Timestep
    influences: jax.Array


class LDQN(DQN):
    """Language-augmented DQN"""
    hparams: LDQNHParams = struct.field(pytree_node=False)

    def init(self, timestep: Timestep, *, key: KeyArray) -> LDQNState:
        hparams = self.hparams
        assert isinstance(hparams.action_space, Discrete)
        iteration = jnp.asarray(0)
        params = self.critic.init(
            key, hparams.obs_space.sample(key)
        )
        params_target = jtu.tree_map(lambda x: x, params)  # copy params
        buffer = RingBuffer.create(
            timestep,
            hparams.replay_memory_size,
            hparams.n_steps,
        )
        annotation_buffer = RingBuffer.create(
            Annotation(timestep, jnp.zeros((len(timestep.action),))),
            hparams.replay_memory_size,
            hparams.n_steps,
        )
        opt_state = self.optimiser.init(params)
        return LDQNState(
            iteration=iteration,
            params=params,
            params_target=params_target,
            opt_state=opt_state,
            buffer=buffer,
            annotation_buffer=annotation_buffer,
            log=DQNLog(),
        )

    def update(
        self,
        train_state: DQNState,
        transition: Timestep,
        *,
        key: KeyArray,
    ) -> DQNState:
        train_state = self.dqn_update(train_state, transition, key=key)
        train_state = self.llm_update(train_state, key=key)
        return train_state

    def llm_update(self, train_state: DQNState, *, key: KeyArray):
        def _sgd_step(params, opt_state):
            loss_fn = lambda params, trans: jnp.mean(
                jax.vmap(self.loss, in_axes=(None, 0, None))(
                    params, trans, train_state.params_target
                )
            )
            loss, grads = jax.value_and_grad(loss_fn)(params, transitions)
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        transitions = train_state.buffer.sample(
            key=key,
            n=self.hparams.llm_batch_size // self.hparams.n_steps * self.hparams.n_llms,
        )
        observations = transitions

        rewards = ask_intrinsic_reward(transitions.observations)

        return train_state

    def dqn_update(
        self,
        train_state: DQNState,
        transition: Timestep,
        *,
        key: KeyArray,
    ) -> DQNState:
        # update iteration
        iteration = jnp.asarray(train_state.iteration + 1, dtype=jnp.int32)

        # update memory
        buffer = train_state.buffer.add(transition)

        transitions = buffer.sample(key, self.hparams.batch_size)

        # update critic
        @jax.jit
        def _sgd_step(params, opt_state):
            loss_fn = lambda params, trans: jnp.mean(
                jax.vmap(self.loss, in_axes=(None, 0, None))(
                    params, trans, train_state.params_target
                )
            )
            loss, grads = jax.value_and_grad(loss_fn)(params, transitions)
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        cond = buffer.size() < self.hparams.replay_memory_size
        cond = jnp.logical_or(cond, iteration % self.hparams.update_frequency != 0)
        params, opt_state, loss = jax.lax.cond(
            cond,
            lambda p, o: _sgd_step(p, o),
            lambda p, o: (p, o, jnp.asarray(float("inf"))),
            train_state.params,
            train_state.opt_state,
        )

        # update target critic
        params_target = optax.periodic_update(
            params,
            train_state.params_target,
            jnp.asarray(iteration, dtype=jnp.int32),
            self.hparams.target_network_update_frequency,
        )

        # log
        log = DQNLog(
            iteration=iteration,
            critic_loss=loss,
            step_type=transition.step_type[-1],
            returns=train_state.log.returns
            + jnp.sum(
                self.hparams.discount ** transition.t[:-1] * transition.reward[:-1]
            ),
            buffer_size=buffer.size(),
        )

        # update train_state
        train_state = train_state.replace(
            iteration=iteration,
            opt_state=opt_state,
            params=params,
            params_target=params_target,
            buffer=buffer,
            log=log,
        )
        return train_state
