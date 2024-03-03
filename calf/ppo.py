from __future__ import annotations
from functools import partial
from typing import Any, Dict, Tuple

import distrax
import flax.linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
from jax import Array
import jax
from jax.random import KeyArray
from jax.lax import stop_gradient
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import rlax

from helx.base.spaces import Discrete
from helx.base.mdp import Timestep, StepType
from helx.envs.environment import Environment

from .trial import run_n_steps, Agent, HParams as HParamsBase


class HParams(HParamsBase):
    discount: float = 0.99
    """The MDP discount factor."""
    lambda_: float = 0.95
    """The lambda parameter for the TD(lambda) algorithm."""
    learning_rate: float = 0.00025
    """The learning rate of the gradient descent algorithm."""
    beta: float = 0.01
    """Entropy bonus coefficient."""
    clip_ratio: float = 0.2
    """The epsilon parameter in the PPO paper."""
    n_actors: int = 2
    """The number of actors to use."""
    n_epochs: int = 4
    """The number of epochs to train for each update."""
    iteration_size: int = 2048
    """The number of steps to collect in total from all environments at update."""
    batch_size: int = 128
    """The number of minibatches to run at each epoch."""
    gradient_clip_norm: float = 0.5
    """The maximum norm of the gradient"""
    advantage_normalisation: bool = False
    """Whether to normalise the advantage function as per
    https://arxiv.org/pdf/2006.05990.pdf"""
    value_clipping: bool = True
    """Whether to clip the value function estimates as per
    https://arxiv.org/pdf/2005.12729.pdf."""
    value_loss_coefficient: float = 0.5
    """The multiplying coefficient for the value loss as per
    https://arxiv.org/pdf/2006.05990.pdf"""


class PPO(Agent):
    """Proximal Policy Optimisation as described
    in https://arxiv.org/abs/1707.06347
    Implementation details as per recommendations
    in https://arxiv.org/pdf/2006.05990.pdf and https://arxiv.org/pdf/2005.12729.pdf
    """

    # static:
    hparams: HParams = struct.field(pytree_node=False)
    optimiser: optax.GradientTransformation = struct.field(pytree_node=False)
    actor: nn.Module = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)
    # dynamic:
    train_state: Dict[str, Any] = struct.field(pytree_node=True)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: KeyArray,
    ) -> PPO:
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

    def policy(self, params: Params, observation: Array) -> distrax.Softmax:
        logits = jnp.asarray(self.actor.apply(params["actor"], observation))
        return distrax.Softmax(logits=logits)

    def value_fn(self, params: Params, observation: Array) -> Array:
        return jnp.asarray(self.critic.apply(params["critic"], observation))[0]

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
        episode_length = self.hparams.iteration_size // self.hparams.n_actors
        return run_n_steps(env, timestep, self, episode_length, key=key)

    def evaluate_experience(self, episodes: Timestep) -> Timestep:
        """Calculate targets for multiple concatenated episodes.
        For episodes with a total sum of T steps, it computes values and advantage
        in the range [0, T-1]"""
        assert (
            episodes.t.ndim == 1
        ), "episodes.t.ndim must be 1, got {} instead.".format(episodes.t.ndim)
        obs = episodes.observation  # s_t \\forall t \\in [0, T]
        reward = episodes.reward[1:]  # r_t \\forall t \\in [0, T-1]
        discount = (episodes.is_mid() * self.hparams.discount**episodes.t)[
            1:
        ]  # \\gamma^t \\forall t \\in [0, T-1]
        value = jax.vmap(self.value_fn, in_axes=(None, 0))(self.params, obs) * (
            episodes.step_type != StepType.TERMINATION
        )  # q(s_t, a'_t) \\forall a'
        advantage = jnp.asarray(
            rlax.truncated_generalized_advantage_estimation(
                reward,
                discount,
                self.hparams.lambda_,
                value,
            )
        )
        episodes.info["value"] = value
        if self.hparams.advantage_normalisation:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        episodes.info["advantage"] = advantage
        # values and advantages from [0, T-1]
        return episodes

    def sample_experience(self, episodes: Timestep, *, key: KeyArray) -> Timestep:
        """Samples a minibatch of transitions from the collected experience with
        the "Shuffle transitions (recompute advantages)" method: see
        https://arxiv.org/pdf/2006.05990.pdf
        Args:
            episodes (Timestep): a Timestep representing a batch of trajectories.
                The first axis is the number of actors, the second axis is time.
            key (Array): a random key to sample actions
        Returns:
            Timestep: a minibatch of transitions (2-steps sars tuples) where the
            first axis is the number of actors * n_steps // n_minibatches
        """
        assert episodes.t.ndim == 2, "episodes.ndim must be 2, got {} instead.".format(
            episodes.t.ndim
        )
        # sample transitions
        batch_size = self.hparams.batch_size
        actor_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.hparams.n_actors
        )
        # exclude last timestep (-1) as it was excluded from the advantage computation
        episode_length = episodes.t.shape[1]
        time_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=episode_length
        )
        transitions_t = episodes[actor_idx, time_idx]  # contains s_{t+1}
        transitions_tp1 = episodes[actor_idx, time_idx + 1]  # contains s_{t+2}
        return jtu.tree_map(
            lambda *x: jnp.stack(x, axis=1), transitions_t, transitions_tp1
        )  # (batch_size, 2)

    def loss(
        self, params: Params, transition: Timestep
    ) -> Tuple[Array, Dict[str, Array]]:
        # make sure the transition has the required info
        assert "advantage" in transition.info
        assert "log_prob" in transition.info
        # stop gradient on advantage and log_prob
        observation = transition.observation[0]  # s_t
        advantage = stop_gradient(transition.info["advantage"][0])  # A(s_t, a_t)
        action_value_old = stop_gradient(transition.info["value"][0])  # v_{k-1}(s_t)
        action = stop_gradient(transition.action[1])  # a_t
        log_prob_old = stop_gradient(transition.info["log_prob"][0])
        # critic loss
        value = self.value_fn(params, observation)  # v(s_t)
        critic_loss = jnp.abs(value - advantage)
        # critic_loss = 0.5 * jnp.square(value - advantage)
        if self.hparams.value_clipping:
            value_clipped = jnp.clip(
                value,
                action_value_old - self.hparams.clip_ratio,
                action_value_old + self.hparams.clip_ratio,
            )
            critic_loss_clipped = 0.5 * jnp.square(value_clipped - advantage)
            critic_loss = jnp.maximum(critic_loss, critic_loss_clipped)
        critic_loss = self.hparams.value_loss_coefficient * critic_loss
        # actor loss
        action_distribution = self.policy(params, observation)
        log_prob = action_distribution.log_prob(action)
        ratio = jnp.exp(log_prob - log_prob_old)
        clipped_ratio = jnp.clip(
            ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio
        )
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        # entropy
        entropy = action_distribution.entropy()
        entropy_loss = -jnp.asarray(entropy) * jnp.asarray(self.hparams.beta)
        # total loss
        loss = actor_loss + critic_loss + entropy_loss
        # logs
        log = {
            "losses/loss": loss,
            "losses/critic_loss": critic_loss,
            "losses/actor_loss": actor_loss,
            "losses/entropy_bonus": entropy_loss,
            "losses/entropy": entropy,
        }
        return loss, log

    @jax.jit
    def update(
        self, episodes: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0))(params, transitions)
            out = jtu.tree_map(lambda x: x.mean(axis=0), out)
            return out

        params, train_state, log = self.params, self.train_state, {}
        for _ in range(self.hparams.n_epochs):
            # calcualte GAE with new (updated) value function and inject in timestep
            episodes = jax.vmap(self.evaluate_experience)(episodes)
            # sample batch of transitions
            transitions = self.sample_experience(episodes, key=key)
            # SGD with PPO loss
            (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
                params, transitions
            )
            opt_state = train_state["opt_state"]
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # update train_state object
            train_state["iteration"] += 1
            train_state["opt_state"] = opt_state
            train_state["params"] = params

            # log gradients norm
            log["losses/grad_norm"] = optax.global_norm(updates)
            if "buffer" in train_state:
                log["buffer_size"] = len(train_state["buffer"])
        return self.replace(train_state=train_state), log
