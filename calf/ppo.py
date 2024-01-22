from __future__ import annotations
from functools import partial
from typing import Any, Dict, Sequence, Tuple

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
    discount: float = 0.99
    """The MDP discount factor."""
    lambda_: float = 0.9
    """The lambda parameter for the TD(lambda) algorithm."""
    learning_rate: float = 0.00025
    """The learning rate of the gradient descent algorithm."""
    beta: float = 0.01
    """Entropy bonus coefficient."""
    clip_ratio: float = 0.2
    """The epsilon parameter in the PPO paper."""
    n_actors: int = 2
    """The number of actors to use."""
    n_epochs: int = 10
    """The number of epochs to train for each update."""
    iteratione_size: int = 1024
    """The number of steps to collect in total from all environments at update."""
    batch_size: int = 256
    """The number of minibatches to run at each epoch."""
    gradient_clip_norm: float = 0.5
    """The maximum norm of the gradient"""
    normalise_advantage: bool = False
    """Whether to normalise the advantage function."""
    value_clipping: bool = False
    """Whether to clip the value function estimates."""


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
        backbone: nn.Module,
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
                backbone.clone(),
                nn.Dense(env.action_space.maximum, kernel_init=rescaled_lecun_normal()),
            ]
        )
        critic = nn.Sequential([backbone.clone(), nn.Dense(env.action_space.maximum)])
        unbatched_obs_sample = env.observation_space.sample(key)[0]
        params_actor = actor.init(key, unbatched_obs_sample)
        params_critic = critic.init(key, unbatched_obs_sample)
        params = {"actor": params_actor, "critic": params_critic}
        optimiser = optax.chain(
            optax.clip_by_global_norm(hparams.gradient_clip_norm),
            optax.scale_by_adam(),
            optax.scale(-hparams.learning_rate),  # negative to minimise
        )
        opt_state = optimiser.init(params)
        return cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            params=params,
            opt_state=opt_state,
        )

    def policy(self, params: Params, observation: Array) -> distrax.Softmax:
        logits = jnp.asarray(self.actor.apply(params["actor"], observation))
        return distrax.Softmax(logits=logits)

    def value_fn(self, params: Params, observation: Array) -> Array:
        return jnp.asarray(self.critic.apply(params["critic"], observation))

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

        @jax.jit
        @partial(jax.vmap, in_axes=(None, 0, 0))
        def batch_policy(params, observation, key):
            action_distribution = self.policy(params, observation)
            action = action_distribution.sample(
                seed=key, sample_shape=env.action_space.shape[1:]
            )
            log_prob = jnp.log(action_distribution.probs)
            return action, log_prob

        # collect trajectories
        episode_length = self.hparams.iteratione_size // self.hparams.n_actors
        episodes = []
        for t in range(episode_length):
            k1, k2, key = jax.random.split(key, num=3)
            k1 = jax.random.split(k1, num=env.action_space.shape[0])
            action, log_prob = batch_policy(self.params, timestep.observation, k1)
            timestep.info["log_prob"] = log_prob
            episodes.append(timestep)
            # cache return
            return_ = timestep.info["return"] * (timestep.is_mid())
            # step the environment
            timestep = env.step(k2, timestep, action)
            # update return
            timestep.info["return"] = return_ + (
                timestep.reward * self.hparams.discount**timestep.t
            )
        # first axis is the number of actors, second axis is time
        trajectories = jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episodes)

        # update current return
        return trajectories, timestep

    def sample_experience(
        self, episodes: Timestep, timestep: Timestep, *, key: KeyArray
    ) -> Timestep:
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
        assert "log_prob" in episodes.info

        batch_size = self.hparams.batch_size
        episode_length = episodes.t.shape[1]
        # compute the advantage for each timestep
        action_values = jax.vmap(
            jax.vmap(self.value_fn, in_axes=(None, 0)), in_axes=(None, 0)
        )(self.params, episodes.observation)
        log_prob = episodes.info["log_prob"]
        state_values = jnp.sum(log_prob * action_values, axis=-1)
        # apply termination mask
        values = state_values * (episodes.step_type != StepType.TERMINATION)
        advantages = jax.vmap(
            rlax.truncated_generalized_advantage_estimation, in_axes=(0, 0, None, 0)
        )(
            episodes.reward[:, :-1],
            self.hparams.discount ** episodes.t[:, :-1],
            self.hparams.lambda_,
            values,
        )
        if self.hparams.normalise_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        episodes.info["advantage"] = advantages
        # sample transitions
        actor_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.hparams.n_actors
        )
        # exclude last timestep (-1) as it was excluded from the advantage computation
        time_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=episode_length - 1
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
        log_prob_old = jax.lax.stop_gradient(
            transition.info["log_prob"][transition.action]
        )
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
        actor_loss = jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        actor_loss = -actor_loss  # maximise
        # entropy
        entropy = jax.lax.stop_gradient(action_distribution.entropy())
        entropy_loss = jnp.asarray(entropy) * jnp.asarray(self.hparams.beta)
        entropy_loss = -entropy_loss  # maximise
        # total loss
        loss = actor_loss + critic_loss + entropy_loss
        # logs
        log = {
            "losses/loss": loss,
            "losses/critic_loss": critic_loss,
            "losses/actor_loss": actor_loss,
            "losses/entropy_bonus": entropy_loss,
            "policy/entropy": entropy,
        }
        return loss, log

    @jax.jit
    def update(
        self, trajectories: Timestep, timestep: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0))(params, transitions)
            out = jtu.tree_map(lambda x: x.mean(axis=0), out)
            return out

        params, opt_state, log = self.params, self.opt_state, {}
        for _ in range(self.hparams.n_epochs):
            transitions = self.sample_experience(trajectories, timestep, key=key)
            (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
                params, transitions
            )
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            log["grads/grad_norm"] = optax.global_norm(updates)
        return self.replace(params=params, opt_state=opt_state), log


def run_n_steps(
    env: Environment, timestep: Timestep, agent: PPO, n_steps: int, *, key: KeyArray
) -> Timestep:
    """Runs `n_steps` in the environment using the agent's policy and returns a
    partial trajectory.

    Args:
        env (Environment): the environment to step in.
        timestep (Timestep): the timestep to start from.
        agent (PPO): the agent.
        n_steps (int): the number of steps to run.
        key (KeyArray): a random key to sample actions.

    Returns:
        Timestep: a partial trajectory of length `n_steps`.
        The trajectory can contain multiple episodes.
        Each timestep in the trajectory has the following structure:
        $(t, o_t, a_{t-1}, r_{t-1}, step_type_{t-1}, info_{t-1})$
        where:
        $a_{t} \\sim \\pi(s_t)$ is the action sampled from the policy conditioned on
        $s_t$; $r_{t} = R(s_t, a_t)$ is the reward after taking the action $a_t$ in
        $s_t$.

    Example:
        (0, s_0, -1,  0.0,      0, info_0)  RESET
        (1, s_1, a_0, r_0, term_0, info_0)
        (2, s_2, a_1, r_1, term_1, info_1)
        (3, s_3, a_2, r_2, term_2, info_2)  TERMINATION
        (0, s_0, -1,  0.0,      0, info_3)  RESET
        (1, s_1, a_0, r_0, term_0, info_0)
        (2, s_2, a_1, r_1, term_1, info_1)  TERMINATION
        (0, s_0, -1,  0.0,      0, info_2)  RESET
    """

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def policy_sample(params, observation, key):
        action_distribution = agent.policy(params, observation)
        action = action_distribution.sample(
            seed=key, sample_shape=env.action_space.shape[1:]
        )
        log_probs = jnp.log(action_distribution.probs)
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
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
                + (next_timestep.reward * agent.hparams.discount**next_timestep.t)
            )
        timestep = next_timestep

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)


def run_episode(env: Environment, agent: PPO, *, key: KeyArray) -> Timestep:
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
                + (next_timestep.reward * agent.hparams.discount**next_timestep.t)
            )
        final = jnp.logical_or(final, timestep.is_last())
        if final.all():
            break
        timestep = next_timestep

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)


def truncated_gae(
    episode: Timestep, values: Array, discount: float, lambda_: float
) -> Array:
    """Computes the truncated Generalised Advantage Estimation (GAE) for a continuos
    stream of experience that may include multiple episodes.
    Args:
        episode (Timestep): a Timestep representing a trajectory of length `T`.
        values (Array): the value function estimates in the range `[0, T]`.
        discount (float): the MDP discount factor.
        lambda_ (float): the lambda parameter for the TD(lambda) algorithm.
    Returns:
        Array: the GAE estimates in the range `[0, T]`."""
    advantage = rlax.truncated_generalized_advantage_estimation(
        episode.reward[:-1],
        episode.is_mid()[:-1] * discount ** episode.t[:-1],
        lambda_,
        values,
    )
    return jnp.asarray(advantage * episode.is_mid())


def rescaled_lecun_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = jnp.float_,
) -> nn.initializers.Initializer:
    return nn.initializers.variance_scaling(
        0.01,
        "fan_in",
        "truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )
