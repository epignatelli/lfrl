from __future__ import annotations
from typing import List, Tuple

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
from helx.base.mdp import Timestep
from helx.envs.environment import Environment


class HParams(struct.PyTreeNode):
    # rl
    discount: float = 0.99
    n_steps: int = 1
    lambda_: float = 0.9
    # sgd
    batch_size: int = 32
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    # buffer
    buffer_capacity: int = 100_000
    # ppo
    """Entropy bonus coefficient."""
    beta: float = 0.01
    """The epsilon parameter in the PPO paper."""
    clip_ratio: float = 0.2
    """The number of actors to use."""
    n_actors: int = 16
    """The number of epochs to train for each update."""
    n_epochs: int = 4
    """The number of minibatches to run at each epoch."""
    n_minibatches: int = 4


class Log(struct.PyTreeNode):
    critic_loss: Array = jnp.array(0.0)
    actor_loss: Array = jnp.array(0.0)
    loss: Array = jnp.array(0.0)
    actor_entropy: Array = jnp.array(0.0)
    returns: Array = jnp.array(0.0)


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
    iteration: Array = struct.field(pytree_node=True)
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
        action_space = env.action_space
        assert isinstance(action_space, Discrete)
        actor = nn.Sequential([backbone, nn.Dense(action_space.maximum)])
        critic = nn.Sequential([backbone, nn.Dense(action_space.maximum)])
        assert isinstance(env.action_space, Discrete)
        iteration = jnp.asarray(0)
        params = actor.init(key, env.observation_space.sample(key))
        opt_state = optimiser.init(params)

        placeholder = jnp.array(0)
        return cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            params=params,
            iteration=iteration,
            opt_state=opt_state,
        )

    @jax.jit
    def policy(self, params: Params, observation: Array) -> distrax.Distribution:
        logits = jnp.asarray(self.actor.apply(params, observation))
        return distrax.Categorical(logits=logits)

    @jax.jit
    def value_fn(self, params: Params, observation: Array) -> Array:
        return jnp.asarray(self.critic.apply(params, observation))

    def collect_experience(self, env: Environment, *, key: KeyArray) -> Timestep:
        """Collects `n_actors` trajectories of experience of length `n_steps`
        from the environment. This method is the only one that interacts with the
        environment, and cannot be jitted unless the environment is JAX-compatible.
        Args:
            env (Environment): the environment must be a batched environment,
                such that `env.step` returns `n_actors` timesteps
            key (Array): a random key to sample actions
        Returns:
            Timestep: a Timestep representing a batch of trajectories.
            The first axis is the number of actors, the second axis is time.
        """
        timestep = env.reset(key)  # this is a batch of timesteps of t=0
        episodes = []
        for t in range(self.hparams.n_steps):
            _, key = jax.random.split(key)
            action_distribution = self.policy(self.params, timestep.observation)
            action, log_probs = action_distribution.sampl_n_and_log_prob(
                seed=key, n=self.hparams.n_actors
            )
            timestep = timestep.replace(info={"log_probs": log_probs})
            episodes.append(timestep)
            timestep = env.step(key, timestep, action)  # get new timestep
        episodes.append(timestep)  # add last timestep
        # first axis is the number of actors, second axis is time
        return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episodes)

    def sample_experience(self, episodes: Timestep, *, key: KeyArray) -> Timestep:
        """Samples a minibatch of transitions from the collected experience, as
        per https://arxiv.org/pdf/2006.05990.pdf, Section 3.5.
        We recalt
        Args:
            episodes (Timestep): a Timestep representing a batch of trajectories.
                The first axis is the number of actors, the second axis is time.
            key (Array): a random key to sample actions
        Returns:
            Timestep: a minibatch of transitions (2-steps sars tuples) where the
            first axis is the number of actors * n_steps // n_minibatches
        """
        # decompose trajectory into transitions
        batch_size = (
            self.hparams.n_actors * self.hparams.n_steps // self.hparams.n_minibatches
        )
        batch_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.hparams.n_actors
        )
        time_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.hparams.n_steps
        )
        transitions = episodes[batch_idx, time_idx : time_idx + self.hparams.n_steps]
        return transitions

    def loss(self, params: Params, transitions: Timestep) -> Tuple[Array, Log]:
        values = self.value_fn(params, transitions.observation[:, -1])
        advantage = jnp.asarray(
            rlax.truncated_generalized_advantage_estimation(
                transitions.reward,
                self.hparams.discount**transitions.t,
                self.hparams.lambda_,
                values,
            )
        )
        # critic loss
        critic_loss = jnp.square(values - advantage) * 0.5
        # actor loss
        log_prob = self.policy(params, transitions.observation).log_prob(
            transitions.action
        )
        ratio = jnp.exp(log_prob - transitions.info["log_probs"])
        clipped_ratio = jnp.clip(
            ratio, 1 / (1 + self.hparams.clip_ratio), 1 + self.hparams.clip_ratio
        )
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        # entropy
        entropy = self.policy(params, transitions.observation).entropy()
        # total loss
        loss = jnp.mean(actor_loss + self.hparams.beta * entropy + critic_loss)
        return loss, Log(critic_loss, actor_loss, loss, entropy, advantage)

    @jax.jit
    def update(self, trajectories: Timestep, *, key: KeyArray) -> Tuple[PPO, List[Log]]:
        params = self.params
        opt_state = self.opt_state
        logs = [Log()] * self.hparams.n_epochs
        for e in range(self.hparams.n_epochs):
            # sample batch
            transitions = self.sample_experience(trajectories, key=key)
            # compute gradients
            (_, log), grads = jax.value_and_grad(self.loss, has_aux=True)(
                params, transitions
            )
            # update params
            updates, opt_state = self.optimiser.update(grads, self.opt_state)
            params = optax.apply_updates(params, updates)
            logs[e] = log

        return self.replace(params=params, opt_state=opt_state), logs
