from __future__ import annotations
from functools import partial
from typing import Dict, Tuple

import distrax
import flax.linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
from jax import Array
import jax
from jax.random import KeyArray
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import rlax

from helx.base.spaces import Discrete
from helx.base.mdp import Timestep, StepType
from helx.envs.environment import Environment


class HParams(struct.PyTreeNode):
    # rl
    discount: float = 0.99
    n_steps: int = 10
    lambda_: float = 0.9
    # sgd
    learning_rate: float = 0.00025
    # buffer
    buffer_capacity: int = 100_000
    # ppo
    """Entropy bonus coefficient."""
    beta: float = 0.01
    """The epsilon parameter in the PPO paper."""
    clip_ratio: float = 0.2
    """The number of actors to use."""
    n_actors: int = 8
    """The number of epochs to train for each update."""
    n_epochs: int = 4
    """The number of steps to collect in total from all environments at update."""
    iteratione_size: int = 256
    """The number of minibatches to run at each epoch."""
    batch_size: int = 4


class Log(struct.PyTreeNode):
    critic_loss: Array = jnp.array(0.0)
    actor_loss: Array = jnp.array(0.0)
    loss: Array = jnp.array(0.0)
    actor_entropy: Array = jnp.array(0.0)
    returns: Array = jnp.array(0.0)
    iteration: Array = jnp.array(0)


class PPO(struct.PyTreeNode):
    """Proximal Policy Optimisation as described
    in https://arxiv.org/abs/1707.06347
    Implementation details as per recommendations
    in https://arxiv.org/pdf/2006.05990.pdf
    """

    # static:
    hparams: HParams = struct.field(pytree_node=False)
    optimiser: optax.GradientTransformation = struct.field(pytree_node=False)
    actor: nn.Module = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)
    # dynamic:
    params: Params = struct.field(pytree_node=True)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        optimiser: optax.GradientTransformation,
        backbone: nn.Module,
        *,
        key: KeyArray,
    ) -> PPO:
        assert isinstance(env.action_space, Discrete)
        # TODO: implement this check
        # we do not check for the number of actors here, and we assume that the
        # environment is batched, such that `env.step` returns `n_actors` timesteps
        # assert env.n_parallel == hparams.n_actors
        actor = nn.Sequential([backbone, nn.Dense(env.action_space.maximum + 1)])
        critic = nn.Sequential([backbone, nn.Dense(env.action_space.maximum + 1)])
        unbatched_obs_sample = env.observation_space.sample(key)[0]
        params_actor = actor.init(key, unbatched_obs_sample)
        params_critic = critic.init(key, unbatched_obs_sample)
        params = {"actor": params_actor, "critic": params_critic}
        opt_state = optimiser.init(params)
        return cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            params=params,
            opt_state=opt_state,
        )

    def policy(self, params: Params, observation: Array) -> distrax.Distribution:
        logits = jnp.asarray(self.actor.apply(params["actor"], observation))
        return distrax.Categorical(logits=logits)

    def value_fn(self, params: Params, observation: Array) -> Array:
        return jnp.asarray(self.critic.apply(params["critic"], observation))

    def collect_experience(self, env: Environment, *, key: KeyArray) -> Timestep:
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

        @jax.jit
        def batch_policy(params, observation, key):
            action_distribution = jax.vmap(self.policy, in_axes=(None, 0))(
                self.params, timestep.observation
            )
            action, log_prob = action_distribution.sample_and_log_prob(
                seed=key, sample_shape=env.action_space.shape[1:]
            )
            return action, log_prob

        timestep = env.reset(key)  # this is a batch of timesteps of t=0
        episode_length = self.hparams.iteratione_size // self.hparams.n_actors
        action, log_prob = batch_policy(self.params, timestep.observation, key)
        timestep.info["log_prob"] = log_prob
        episodes = [timestep]
        for t in range(episode_length - 1):
            k1, key = jax.random.split(key)
            timestep = env.step(key, timestep, action)  # potentiall jax-incompat
            action, log_prob = batch_policy(self.params, timestep.observation, k1)
            timestep.info["log_prob"] = log_prob
            episodes.append(timestep)
        # first axis is the number of actors, second axis is time
        return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episodes)

    def sample_experience(self, episodes: Timestep, *, key: KeyArray) -> Timestep:
        """Samples a minibatch of transitions from the collected experience using
        the "Shuffle transitions (recompute advantages)" method from
        https://arxiv.org/pdf/2006.05990.pdf
        Args:
            episodes (Timestep): a Timestep representing a batch of trajectories.
                The first axis is the number of actors, the second axis is time.
            key (Array): a random key to sample actions
        Returns:
            Timestep: a minibatch of transitions (2-steps sars tuples) where the
            first axis is the number of actors * n_steps // n_minibatches
        """
        msg = "`episodes` must be a batch of trajectories with at least two-steps,\
        got `episode.t.ndim` = {} instead".format(
            episodes.t.ndim
        )
        assert episodes.t.ndim == 2, msg
        batch_size = self.hparams.batch_size
        episode_length = self.hparams.iteratione_size // self.hparams.n_actors
        # compute the advantage for each timestep
        action_values = jax.vmap(
            jax.vmap(self.value_fn, in_axes=(None, 0)), in_axes=(None, 0)
        )(self.params, episodes.observation)
        state_values = jnp.mean(action_values, axis=-1)
        # apply termination mask
        values = state_values * (episodes.step_type != StepType.TERMINATION)
        advantages = jax.vmap(
            rlax.truncated_generalized_advantage_estimation, in_axes=(0, 0, None, 0)
        )(
            episodes.reward[:, 1:],
            self.hparams.discount ** episodes.t[:, 1:],
            self.hparams.lambda_,
            values,
        )
        episodes.info["advantage"] = advantages
        # sample transitions
        actor_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.hparams.n_actors
        )
        time_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=episode_length
        )
        transitions = episodes[actor_idx, time_idx]  # (batch_size,)
        return transitions

    def loss(
        self, params: Params, transition: Timestep
    ) -> Tuple[Array, Dict[str, Array]]:
        # make sure the transition has the required info
        assert "advantage" in transition.info
        assert "log_prob" in transition.info
        # stop gradient on advantage and log_prob
        advantage = transition.info["advantage"]
        advantage = jax.lax.stop_gradient(advantage)
        log_prob_old = jax.lax.stop_gradient(transition.info["log_prob"])
        # critic loss
        q_values = self.value_fn(params, transition.observation)
        critic_loss = 0.5 * jnp.square(q_values[transition.action] - advantage)
        # actor loss
        action_distribution = self.policy(params, transition.observation)
        log_prob = action_distribution.log_prob(transition.action)
        ratio = jnp.exp(log_prob - log_prob_old)
        clipped_ratio = jnp.clip(
            ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio
        )
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        # entropy
        entropy = jax.lax.stop_gradient(action_distribution.entropy())
        entropy_bonus = - jnp.asarray(self.hparams.beta) * entropy
        # total loss
        loss = actor_loss + entropy_bonus + critic_loss
        # logs
        log = {
            "loss": loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "entropy": entropy,
        }
        return loss, log

    # @jax.jit
    def update(
        self, trajectories: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0))(params, transitions)
            out = jtu.tree_map(lambda x: x.mean(axis=0), out)
            return out

        params, opt_state, log = self.params, self.opt_state, {}
        for _ in range(self.hparams.n_epochs):
            transitions = self.sample_experience(trajectories, key=key)
            (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
                params, transitions
            )
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        return self.replace(params=params, opt_state=opt_state), log
